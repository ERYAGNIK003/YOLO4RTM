# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""
from mmcv.ops import box_iou_rotated
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
from utils.rboxs_utils import rbox2poly
import math

#https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/models/losses/gaussian_dist_loss_v1.py
def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    #wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e4).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    sigma = R.bmm(S.square()).bmm(R.permute(0, 2, 1)).reshape(_shape[:-1] + (2, 2))
    
    return xy, sigma

def bcd_loss(pred, target, fun='log1p', tau=1.0):
    mu_p, sigma_p = pred
    mu_t, sigma_t = target

    mu_p = mu_p.reshape(-1, 2).float()
    mu_t = mu_t.reshape(-1, 2).float()
    sigma_p = sigma_p.reshape(-1, 2, 2).float()
    sigma_t = sigma_t.reshape(-1, 2, 2).float()

    eps = 1e-6
    I = torch.eye(2, device=mu_p.device).unsqueeze(0)

    sigma_p = sigma_p + eps * I
    sigma_t = sigma_t + eps * I

    delta = (mu_p - mu_t).unsqueeze(-1)

    sigma = 0.5 * (sigma_p + sigma_t)
    sigma = sigma + eps * I

    # Stable determinants
    det_sigma = torch.det(sigma).clamp(min=eps)
    det_sp = torch.det(sigma_p).clamp(min=eps)
    det_st = torch.det(sigma_t).clamp(min=eps)

    term1 = torch.log(det_sigma / torch.sqrt(det_sp * det_st)).reshape(-1, 1)

    # Stable solve instead of inverse
    term2 = delta.transpose(-1, -2).matmul(
        torch.linalg.solve(sigma, delta)
    ).squeeze(-1)

    dis = 0.5 * term1 + 0.125 * term2
    bcd_dis = dis.clamp(min=eps)

    if fun == 'sqrt':
        loss = 1 - 1 / (tau + torch.sqrt(bcd_dis))
    elif fun == 'log1p':
        loss = 1 - 1 / (tau + torch.log1p(bcd_dis))
    else:
        loss = 1 - 1 / (tau + bcd_dis)

    return loss

def kld_loss(pred, target, fun='sqrt', tau=1.0):

    mu_p, sigma_p = pred
    mu_t, sigma_t = target

    mu_p = mu_p.reshape(-1,2).float()
    mu_t = mu_t.reshape(-1,2).float()
    sigma_p = sigma_p.reshape(-1,2,2).float()
    sigma_t = sigma_t.reshape(-1,2,2).float()

    eps = 1e-6
    eye = torch.eye(2, device=sigma_p.device).unsqueeze(0)

    sigma_p = sigma_p + eps * eye
    sigma_t = sigma_t + eps * eye

    sigma_t_inv = torch.linalg.inv(sigma_t)
    
    delta = (mu_p - mu_t).unsqueeze(-1)

    term1 = (delta.transpose(-1,-2) @ sigma_t_inv @ delta).squeeze(-1)
    
    trace_term = torch.diagonal(
        sigma_t_inv @ sigma_p,
        dim1=-2,
        dim2=-1
    ).sum(-1, keepdim=True)
    
    
    sign_p, log_det_sigma_p = torch.linalg.slogdet(sigma_p)
    sign_t, log_det_sigma_t = torch.linalg.slogdet(sigma_t)
    
    log_term = (log_det_sigma_t - log_det_sigma_p).reshape(-1,1)

    dis = term1 + trace_term + log_term - 2

    kl_dis = torch.clamp(dis, min=1e-6)
    

    if fun == 'sqrt':
        kl_loss = 1 - 1/(tau + torch.sqrt(kl_dis))
    else:
        kl_loss = 1 - 1/(tau + torch.log1p(kl_dis))

    return kl_loss

def gwd_loss(pred, target, fun='log1p', tau=1.0):
    """Gaussian Wasserstein distance loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    mu_p, sigma_p = pred
    mu_t, sigma_t = target

    mu_p = mu_p.reshape(-1,2).float()
    mu_t = mu_t.reshape(-1,2).float()
    sigma_p = sigma_p.reshape(-1,2,2).float()
    sigma_t = sigma_t.reshape(-1,2,2).float()

    eps = 1e-6
    I = torch.eye(2, device=mu_p.device).unsqueeze(0)

    sigma_p = sigma_p + eps * I
    sigma_t = sigma_t + eps * I
    
    xy_distance = (mu_p - mu_t).square().sum(dim=-1)

    whr_distance = sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (sigma_p.bmm(sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (sigma_p.det() * sigma_t.det()).clamp(0).sqrt()
    whr_distance += (-2) * (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt()

    dis = xy_distance + whr_distance
    gwd_dis = dis.clamp(min=1e-6)

    if fun == 'sqrt':
        loss = 1 - 1 / (tau + torch.sqrt(gwd_dis))
    elif fun == 'log1p':
        loss = 1 - 1 / (tau + torch.log1p(gwd_dis))
    else:
        scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
        loss = torch.log1p(torch.sqrt(gwd_dis) / scale)
    return loss

def kfiou_loss(pred,
               target,
               pred_decode=None,
               targets_decode=None,
               fun=None,
               beta=1.0 / 9.0,
               eps=1e-6):
    """Kalman filter IoU loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        pred_decode (torch.Tensor): Predicted decode bboxes.
        targets_decode (torch.Tensor): Corresponding gt decode bboxes.
        fun (str): The function applied to distance. Defaults to None.
        beta (float): Defaults to 1.0/9.0.
        eps (float): Defaults to 1e-6.

    Returns:
        loss (torch.Tensor)
    """
    xy_p = pred[:, :2]
    xy_t = target[:, :2]
    _, Sigma_p = xy_wh_r_2_xy_sigma(pred_decode)
    _, Sigma_t = xy_wh_r_2_xy_sigma(targets_decode)

    # Smooth-L1 norm
    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)
    Vb_p = 4 * Sigma_p.det().sqrt()
    Vb_t = 4 * Sigma_t.det().sqrt()
    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
    Sigma = Sigma_p - K.bmm(Sigma_p)
    Vb = 4 * Sigma.det().sqrt()
    Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - KFIoU

    loss = (xy_loss + kf_loss).clamp(0)

    return loss


def gauss_loss(boxes1, boxes2, distance_type="kld"):
    riou=box_iou_rotated(boxes1, boxes2, aligned=True, clockwise=True)
    pred=xy_wh_r_2_xy_sigma(boxes1)
    target=xy_wh_r_2_xy_sigma(boxes2)
    g_loss=0
    if distance_type=="kld":
        g_loss=kld_loss(pred, target)
    elif distance_type=="bcd":
        g_loss=bcd_loss(pred, target)
    elif distance_type=="kfiou":
        g_loss=kfiou_loss(pred, target,pred_decode=boxes1,targets_decode=boxes2)
    else:
        g_loss=gwd_loss(pred, target)
        
    return g_loss,riou

def rDIoU(boxes1,boxes2,eps=1e-7):
    rdiou=0.0
    riou=box_iou_rotated(boxes1, boxes2, aligned=True, clockwise=True)

    riou = torch.nan_to_num(riou, nan=0.0, posinf=0.0, neginf=0.0)
                
    poly1= rbox2poly(boxes1) #tensor
    poly2= rbox2poly(boxes2) #tensor
    
    # minimum bounding box of boxes10 and boxes20
    b1_x1, b1_x2 = poly1[:, 0::2].min(dim=1)[0], poly1[:, 0::2].max(dim=1)[0] # n,
    b1_y1, b1_y2 = poly1[:, 1::2].min(dim=1)[0], poly1[:, 1::2].max(dim=1)[0] # n,
    b2_x1, b2_x2 = poly2[:, 0::2].min(dim=1)[0], poly2[:, 0::2].max(dim=1)[0] # n,
    b2_y1, b2_y2 = poly2[:, 1::2].min(dim=1)[0], poly2[:, 1::2].max(dim=1)[0] # n,
    
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared, enclosing_diagonal_squared

    #rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
    dx = boxes1[:, 0] - boxes2[:, 0]
    dy = boxes1[:, 1] - boxes2[:, 1]
    rho2 = dx * dx + dy * dy
    
    # DIoU
    rdiou = riou - (rho2 / c2) #could be negetive when no overlap, hence 1- rdiou--> strong positive
    
    return rdiou, riou

# GIoU https://arxiv.org/pdf/1902.09630.pdf
# Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
def rCIoU(boxes1,boxes2,eps=1e-7):
    
    riou = box_iou_rotated(boxes1, boxes2, aligned=True, clockwise=True)
    #https://github.com/XinzeLee/RotateObjectDetection/blob/main/rotate-yolov5/utils/general.py
    
    poly1= rbox2poly(boxes1) #tensor
    poly2= rbox2poly(boxes2) #tensor
    
    # minimum bounding box of boxes10 and boxes20
    b1_x1, b1_x2 = poly1[:, 0::2].min(dim=1)[0], poly1[:, 0::2].max(dim=1)[0] # n,
    b1_y1, b1_y2 = poly1[:, 1::2].min(dim=1)[0], poly1[:, 1::2].max(dim=1)[0] # n,
    b2_x1, b2_x2 = poly2[:, 0::2].min(dim=1)[0], poly2[:, 0::2].max(dim=1)[0] # n,
    b2_y1, b2_y2 = poly2[:, 1::2].min(dim=1)[0], poly2[:, 1::2].max(dim=1)[0] # n,
    
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
    # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    w2, h2 = b2_x2-b2_x1, b2_y2-b2_y1+eps
    w1, h1 = b1_x2-b1_x1, b1_y2-b1_y1+eps
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - riou + (1 + eps))
    riou -= (rho2 / c2 + v * alpha)  # CIoU
    
    return riou

def fpdiou(pred, target, eps=1e-7):
    """
    pred/target: (N,5)
    format: (x,y,w,h,theta) in radians
    """

    # IoU
    #iou = box_iou_rotated(pred, target, aligned=True, clockwise=True).clamp(min=eps) #higher is better
    iou = bbox_iou(pred[:, :4].T, target[:, :4], x1y1x2y2=False, CIoU=True).clamp(min=eps)
    
    # ----------------------
    # Center distance penalty
    # ----------------------
    center_dist = (pred[:, 0] - target[:, 0])**2 + \
                  (pred[:, 1] - target[:, 1])**2

    # axis-aligned enclosing box (stable approximation)
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    gt_x1 = target[:, 0] - target[:, 2] / 2
    gt_y1 = target[:, 1] - target[:, 3] / 2
    gt_x2 = target[:, 0] + target[:, 2] / 2
    gt_y2 = target[:, 1] + target[:, 3] / 2

    cw = torch.max(pred_x2, gt_x2) - torch.min(pred_x1, gt_x1)
    ch = torch.max(pred_y2, gt_y2) - torch.min(pred_y1, gt_y1)

    c_diag = cw**2 + ch**2 + eps
    center_term = center_dist / c_diag #lower is better

    # ----------------------
    # Angle penalty
    # ----------------------
    ad = angle_diff(pred[:, 4], target[:, 4])
    angle_term = (1 - torch.cos(ad))/2.0 #1- cos(pi)=2 #lower is better

    # FPDIoU metric (higher is better)
    fpdiou_score = iou - 0.5 * center_term - 0.25 * angle_term

    return fpdiou_score.clamp(0, 1),iou

def fpdiou_loss(pred, target, linear=False, mode='linear', eps=1e-6) -> torch.Tensor:
    """Rotated IoU loss.

    Computing the IoU loss between a set of predicted rbboxes and target
     rbboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    from mmcv.ops.diff_iou_rotated import box2corners,diff_iou_rotated_2d
    import numpy as np
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn(
            'DeprecationWarning: Setting "linear=True" in '
            'poly_iou_loss is deprecated, please use "mode=`linear`" '
            'instead.')

    if diff_iou_rotated_2d is None:
        raise ImportError('Please install mmcv-full >= 1.5.0.')
    
    riou = box_iou_rotated(pred, target, aligned=True, clockwise=True)
    
    ious = box_iou_rotated(pred, target, aligned=True, clockwise=True)
    #ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    #Used-with Only flag
    #ious = torch.nan_to_num(ious, nan=0.0, posinf=0.0, neginf=0.0)
    #ious = ious.squeeze(0).clamp(min=eps)

    prd_four_points=box2corners(pred.unsqueeze(0))[0].detach().cpu().numpy()
    gt_four_points=box2corners(target.unsqueeze(0))[0].detach().cpu().numpy()
    res=[]
    for prd in range(len(prd_four_points)):
        prd_coords=prd_four_points[prd]
        prd_coords=np.asarray(sorted(prd_coords,key=lambda x:x[0]))
        gt_coords=gt_four_points[prd]
        gt_coords=np.asarray(sorted(gt_coords,key=lambda x:x[0]))
        #First line changed, see original implementation in mmrotate
        dis= math.pow(prd_coords[0][0] - gt_coords[0][0], 2) + math.pow(prd_coords[0][1] - gt_coords[0][1], 2)+\
             math.pow(prd_coords[1][0] - gt_coords[1][0], 2) + math.pow(prd_coords[1][1] - gt_coords[1][1], 2) +\
             math.pow(prd_coords[2][0] - gt_coords[2][0], 2) + math.pow(prd_coords[2][1] - gt_coords[2][1], 2)+\
             math.pow(prd_coords[3][0] - gt_coords[3][0], 2) + math.pow(prd_coords[3][1] - gt_coords[3][1], 2)
        res.append(dis/(8*1024**2))#changed

    res=torch.tensor(res,device=torch.device('cuda:0'))
    if mode == 'linear':
        loss =1-ious+ res
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss, riou


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

        
class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        #BCEtheta = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['theta_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            #BCEtheta = FocalLoss(BCEtheta, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.stride = det.stride # tensor([8., 16., 32., ...])
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        # self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.ssi = list(self.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        #self.BCEtheta = BCEtheta
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy w h theta_box hx hy])

        Return：
            total_loss * bs (tensor): [1] 
            torch.cat((lbox, lobj, lcls, lhead)).detach(): [4]
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lhead = torch.zeros(1, device=device) # heading angle related loss
        
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # tcls, tbox, indices, anchors, tgaussian_theta = self.build_targets(p, targets)  # targets
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        tcls, trbox, thx, thy, indices, anchors = self.build_targets(p, targets)  # targets
    
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets, (n_targets, self.no)

                # prediction format [x0 y1 w2 h3 theta_box4 hx5 hy6 objectness7 class_probs...]
                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i] # featuremap pixel

                # ptheta_box = (ps[:, 4:5].sigmoid() - 0.5) * math.pi
                ptheta_box = torch.tanh(ps[:, 4:5]) * (math.pi / 2)
                
                pbox = torch.cat((pxy, pwh, ptheta_box), 1)  # predicted box

                # Calculate loss for hx, hy
                phx = ps[:, 5:6]
                phy = ps[:, 6:7]
                eps = 1e-8
                norm = torch.sqrt(phx**2 + phy**2 + eps)
                phx = phx / norm
                phy = phy / norm
                #loss_heading = 1 - (phx * thx[i] + phy * thy[i])
                pred_vec = torch.cat([phx, phy], dim=1)
                gt_vec   = torch.cat([thx[i], thy[i]], dim=1)
                cos_sim = F.cosine_similarity(pred_vec, gt_vec, dim=1)
                lhead += (1.0 - cos_sim).mean()

                #rCIoU
                '''
                riou = rCIoU(pbox, trbox[i])
                
                riou = torch.nan_to_num(riou, nan=0.0, posinf=0.0, neginf=0.0)
                riou = riou.clamp(0.0, 1.0)
                
                lbox += (1.0 - riou).mean()

                lbox += 0.2 * nn.SmoothL1Loss(beta=0.11)(pbox, trbox[i])#help when box loss in 0, increased some time during validation

                '''
                # split
                pxywh = pbox[:, :4]
                txywh = trbox[i][:, :4]
                ttheta_box = trbox[i][:, 4:5]

                #rDIoU loss
                # rdiou, riou = rDIoU(pbox, trbox[i])
                # lbox += (1 - rdiou).mean()

                #rfiou, riou = fpdiou_loss(pbox, trbox[i])             
                #lbox += rfiou.mean()
                
                rgloss, riou = gauss_loss(pbox, trbox[i], distance_type="bcd")#distance_type = gwd, bcd
                lbox += rgloss.mean()
                
                # coordinate stabilizer
                lbox += 0.1 * nn.SmoothL1Loss(beta=0.11)(pxywh, txywh) #Don't use theta_box for l1 loss because of angle discontinuity at boundary

                # better theta loss
                angle_diff = torch.atan2(torch.sin(ptheta_box - ttheta_box),torch.cos(ptheta_box - ttheta_box))
                theta_box_penalty= (1 - torch.cos(angle_diff)) / 2.0
                lbox += 0.1 * theta_box_penalty.mean()
                
                
                # Objectness
                # score_iou = iou.detach().clamp(0).type(tobj.dtype)
                score_iou = riou.detach().clamp(0).type(tobj.dtype)
                # score_iou = rot_score.detach().clamp(0,1).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                 
                # Classification
                class_index = 8 + self.nc
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t = torch.full_like(ps[:, 8:class_index], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                    lcls += self.BCEcls(ps[:, 8:class_index], t)  # BCE
                
                # theta Classification by Circular Smooth Label
                # t_theta = tgaussian_theta[i].type(ps.dtype) # target theta_gaussian_labels
                # ltheta += self.BCEtheta(ps[:, class_index:], t_theta)

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 7], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lhead *= self.hyp['head']
        # ltheta *= self.hyp['theta']
        bs = tobj.shape[0]  # batch size

        # return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
        # return (lbox + lobj + lcls + ltheta) * bs, torch.cat((lbox, lobj, lcls, ltheta)).detach()
        return (lbox + lobj + lcls + lhead) * bs, torch.cat((lbox, lobj, lcls, lhead)).detach()

    def build_targets(self, p, targets):
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            # targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) pixel
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy w h theta_box hx hy])
        Return：non-normalized data
            tcls (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
            tbox (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 4) featuremap pixel
            indices (list[P3_out,...]): len=self.na, tensor.size(4, n_filter2) [b, a, gj, gi]
            anch (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 2)
            # tgaussian_theta (list[P3_out,...]): len=self.na, tensor.size(n_filter2, hyp['cls_theta'])
            # ttheta (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
            thx (list[hx,...]): hx of heading angle
            thy (list[hx,...]): hy of heading angle
        """
        # Build targets for compute_loss()
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, trbox, indices, anch = [], [], [], []
        # ttheta, tgaussian_theta = [], []
        thx = []
        thy = []
        # gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        feature_wh = torch.ones(2, device=targets.device)  # feature_wh
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets (tensor): (n_gt_all_batch, c) -> (na, n_gt_all_batch, c) -> (na, n_gt_all_batch, c+1)
        # targets (tensor): (na, n_gt_all_batch, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index]])
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], # tensor: (5, 2)
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i] 
            # gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain=[1, 1, w, h, w, h, 1, 1]
            feature_wh[0:2] = torch.tensor(p[i].shape)[[3, 2]]  # xyxy gain=[w_f, h_f]

            # Match targets to anchors
            # t = targets * gain # xywh featuremap pixel
            t = targets.clone() # (na, n_gt_all_batch, c+1)
            t[:, :, 2:6] /= self.stride[i] # xyls featuremap pixel
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # edge_ls ratio, torch.size(na, n_gt_all_batch, 2)
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare, torch.size(na, n_gt_all_batch)
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter; Tensor.size(n_filter1, c+1)

                # Offsets
                gxy = t[:, 2:4]  # grid xy; (n_filter1, 2)
                # gxi = gain[[2, 3]] - gxy  # inverse
                gxi = feature_wh[[0, 1]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) # (5, n_filter1)
                t = t.repeat((5, 1, 1))[j] # (n_filter1, c+1) -> (5, n_filter1, c+1) -> (n_filter2, c+1)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # (5, n_filter1, 2) -> (n_filter2, 2)
            else:
                t = targets[0] # (n_gt_all_batch, c+1)
                offsets = 0

            # Define, t (tensor): (n_filter2, [img_index, clsid, cx, cy, l, s, theta_box, hx, hy, anchor_index])
            b, c = t[:, :2].long().T  # image, class; (n_filter2)
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            # theta = t[:, 6]
            # gtheta =  t[:, 6:7] #raw angle value in radians
            # gaussian_theta_labels = t[:, 7:-1]
            gtheta =  t[:, 6:7] # theta_box θ_box∈[-pi/2, pi/2)
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            ghx = t[:, 7:8] # hx of heading angle
            thx.append(ghx)
            ghy = t[:, 8:9] # hy of heading angle
            thy.append(ghy)
            
            # Append
            a = t[:, -1].long()  # anchor indices 取整
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, feature_wh[1] - 1), gi.clamp_(0, feature_wh[0] - 1)))  # image, anchor, grid indices
            trbox.append(torch.cat((gxy - gij, gwh, gtheta), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            # ttheta.append(theta) # theta, θ∈[-pi/2, pi/2)
            # tgaussian_theta.append(gaussian_theta_labels)

        # return tcls, tbox, indices, anch
        # return tcls, tbox, indices, anch, tgaussian_theta #, ttheta
        # return tcls, tbox, indices, anch #now tbox itself contains rotated box
        return tcls, trbox, thx, thy, indices, anch #now tbox itself contains rotated box
