from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
import cv2
import numpy as np
import random
import os
import argparse

degrees= 180
translate= 0.1
scale= 0.5
hsv_h= 0.0188
hsv_s= 0.704
hsv_v= 0.36
'''
YOLO
mosaic: 0.3
mixup: 0.1
'''
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

#label format: (number of gts,[cls, x1,y1,x2,y2,x3,y3,x4,y4, θ])
def read_label(lb_file):
    if os.path.isfile(lb_file):  
        with open(lb_file) as f:
            labels = []
            for line in f.read().strip().splitlines():
                if not line.strip():
                    continue  # skip empty lines
                
                if line.startswith("flightHeight"):
                    continue  # skip metadata
                
                parts = line.split()
                
                if len(parts) < 11:
                    continue  # skip invalid lines safely
                
                labels.append(parts)           
        l_ = []
        for label in labels:
            if label[-1] == "2": # diffcult
                continue
            cls_id = 0 #cls_name_list.index(label[8])
            l_.append(np.concatenate((cls_id, label[:8], label[10]), axis=None))
        l = np.array(l_, dtype=np.float32)
        return l
    else:
        return None

def apply_aug(img, labels, flip_ud=False, flip_lr=False, hsv=False, affine=False):
    nl=len(labels)
    img_h, img_w = img.shape[0], img.shape[1]
    
    #vertical flip: up-down 
    if nl and flip_ud:
        img = np.flipud(img)
        labels[:, 2:-1:2] = img_h - labels[:, 2:-1:2] - 1
        labels[:, -1] = (-labels[:, -1]) % 360

    #horizontal flip: left-right 
    if nl and flip_lr:
        img = np.fliplr(img)
        labels[:, 1:-1:2] = img_w - labels[:, 1:-1:2] - 1
        labels[:, -1] = (180 - labels[:, -1]) % 360

    #color augmentation
    if nl and hsv:
        augment_hsv(img, hgain=hsv_h, sgain=hsv_s, vgain=hsv_v)

    #afine transformation
    if nl and affine:
        angle = random.uniform(-degrees, degrees)  # CCW positive
        scale_factor = random.uniform(1 - scale, 1 + scale)

        tx = random.uniform(-translate, translate) * img_w
        ty = random.uniform(-translate, translate) * img_h

        # -----------------------
        # 2. Build affine matrix
        # -----------------------
        center = (img_w / 2, img_h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale_factor)

        # Add translation
        M[:, 2] += (tx, ty)

        # -----------------------
        # 3. Transform image
        # -----------------------
        img = cv2.warpAffine(
            img,
            M,
            dsize=(img_w, img_h),
            flags=cv2.INTER_LINEAR,
            borderValue=(114, 114, 114),
            )
        # reshape polygon points
        polys = labels[:, 1:9].reshape(-1, 4, 2)

        # add ones for affine
        ones = np.ones((polys.shape[0], 4, 1))
        polys_homo = np.concatenate([polys, ones], axis=2)  # Nx4x3

        # apply affine transform
        polys_transformed = np.matmul(polys_homo, M.T)

        labels[:, 1:9] = polys_transformed.reshape(-1, 8)

        # -----------------------
        # 5. Update heading angle
        # -----------------------
        # since image rotated by `angle` CCW
        # and your theta is CW positive
        labels[:, -1] = (labels[:, -1] - angle) % 360
        
    img = np.ascontiguousarray(img)
    return img, labels

   
def draw_lbl(img, labels, out_file, isGT=False):
    for label in labels:
        #print(label)
        #label format: (number of gts,[cls, x1,y1,x2,y2,x3,y3,x4,y4, θ])        
        color = colors(label[0])
        poly=label[1:9]
        pts = np.array(poly, dtype=np.float32).reshape(4, 2)
        # OpenCV min area rect
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        if w < h:
            w, h = h, w
       
        theta_rad = np.deg2rad(label[9])
        dx = (w / 2) * np.cos(theta_rad)
        dy = (w / 2) * np.sin(theta_rad)
        x2 = int(cx + dx)
        y2 = int(cy + dy)
        cx = int(cx)
        cy = int(cy)
        polygon_list = np.array([(poly[0], poly[1]), (poly[2], poly[3]),(poly[4], poly[5]), (poly[6], poly[7])], np.int32)
        cv2.drawContours(img, contours=[polygon_list], contourIdx=-1, color=color, thickness=2)
        cv2.arrowedLine(img, pt1=(cx,cy), pt2=(x2,y2), color=color, thickness=2, tipLength=0.25)
        if isGT:
            #draw angle at center
            ang_deg=int(label[9])
            cv2.putText(img, f"{ang_deg}", (int(cx), int(cy)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imwrite(out_file,img)
    return



def parse_opt():
    parser = argparse.ArgumentParser()

    # Augmentation switches (all default False)
    parser.add_argument('--flipud', action='store_true', help='enable vertical flip augmentation')
    parser.add_argument('--fliplr', action='store_true', help='enable horizontal flip augmentation')
    parser.add_argument('--hsv', action='store_true', help='enable HSV augmentation')
    parser.add_argument('--affine', action='store_true', help='enable affine augmentation')
    parser.add_argument('--no', action='store_true', help='No augmentation only annotation of ground truth')
    return parser.parse_args()



def main():
    opt = parse_opt()
    fname="PIOp_DIbl_7og8"
    img_file=f"{fname}.jpg"
    lb_file=f"{fname}.txt"
    labels=read_label(lb_file)   
    if labels.shape[0] and not opt.no:
        img=cv2.imread(img_file)
        img,labels=apply_aug(img, labels, flip_ud= opt.flipud, flip_lr= opt.fliplr, hsv=opt.hsv, affine= opt.affine)
        out_file=f"{fname}_aug.jpg"
        draw_lbl(img, labels, out_file)
    elif labels.shape[0] and opt.no:
        img=cv2.imread(img_file)
        out_file=f"{fname}_GT.jpg"
        draw_lbl(img, labels, out_file, isGT=True)
        
if __name__ == "__main__":
    main()
