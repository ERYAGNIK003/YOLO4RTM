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

                rdiou, riou = rDIoU(pbox, trbox[i])
                lbox += (1 - rdiou).mean()

                # coordinate stabilizer
                lbox += 0.1 * nn.SmoothL1Loss(beta=0.11)(pxywh, txywh) #Don't use theta_box for l1 loss because of angle discontinuity at boundary

                # better theta loss
                angle_diff = torch.atan2(torch.sin(ptheta_box - ttheta_box),torch.cos(ptheta_box - ttheta_box))
                theta_box_penalty= (1 - torch.cos(angle_diff)) / 2.0
                lbox += 0.05 * theta_box_penalty.mean()
                
                
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
