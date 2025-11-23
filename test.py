import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
from skimage import io
from PIL import Image
import cv2
import torchvision
import retinaface_model
import os
import eval_model
def pad_to_square(img, pad_value):
    _, h, w = img.shape
    dim_diff = np.abs(h - w)
    
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def get_args():
    parser = argparse.ArgumentParser(description="Detect program for retinaface.")
    parser.add_argument('--image_path', type=str, default='test.jpg', help='Path for image to detect')
    parser.add_argument('--model_path', type=str, help='Path for model')
    parser.add_argument('--save_path', type=str, default='./out', help='Path for result image')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--scale', type=float, default=1.0, help='Image resize scale', )
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    #Tạo model
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    RetinaFace = retinaface_model.create_retinaface(return_layers)

    # Load trained model
    retina_dict = RetinaFace.state_dict()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_state_dict = torch.load(args.model_path, map_location=device)
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    RetinaFace.load_state_dict(pretrained_dict)

    RetinaFace = RetinaFace.to(device)
    RetinaFace.eval()

    # Read image
    img = skimage.io.imread(args.image_path)
    img = torch.from_numpy(img)
    img = img.permute(2,0,1)

    if not args.scale == 1.0:
        size1 = int(img.shape[1]/args.scale)
        size2 = int(img.shape[2]/args.scale)
        img = resize(img.float(),(size1,size2))

    input_img = img.unsqueeze(0).float().to(device)
    picked_boxes, picked_landmarks, picked_scores = eval_model.get_detections(input_img, RetinaFace, score_threshold=0.5, iou_threshold=0.3)

    
    np_img = img.cpu().permute(1,2,0).numpy()
    np_img.astype(int)
    img = cv2.cvtColor(np_img.astype(np.uint8),cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for j, boxes in enumerate(picked_boxes):
        if boxes is not None:
            for box, landmark, score in zip(boxes,picked_landmarks[j],picked_scores[j]):
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    image_name = args.image_path.split('/')[-1]
    save_path = os.path.join(args.save_path,image_name)
    cv2.imwrite(save_path, img)
    cv2.imshow('RetinaFace-Pytorch',img)
    cv2.waitKey()

if __name__=='__main__':
    main()