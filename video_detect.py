#-*- coding: UTF-8 -*-
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
import eval_model
import retinaface_model
import os

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def get_args():
    parser = argparse.ArgumentParser(description="Detect program for retinaface.")
    parser.add_argument('--video_path', type=int, default=0, help='Path for image to detect')
    parser.add_argument('--model_path', type=str,default='app/Retinaface/model_epoch_3.pt', help='Path for model')
    parser.add_argument('--save_path', type=str, default='./out/result.avi', help='Path for result image')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--scale', type=float, default=1.0, help='Image resize scale', )
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    RetinaFace = retinaface_model.create_retinaface(return_layers)

    # Tải model
    retina_dict = RetinaFace.state_dict()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_state_dict = torch.load(args.model_path, map_location=device)
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    RetinaFace.load_state_dict(pretrained_dict)

    RetinaFace = RetinaFace.to(device)
    RetinaFace.eval()

    # Read video
    cap = cv2.VideoCapture(args.video_path)

    codec = cv2.VideoWriter_fourcc(*'MJPG')

    width = int(cap.get(3))
    height = int(cap.get(4))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = 25.0

    out = cv2.VideoWriter('args.save_path', codec, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX

    while(True):
        ret, img = cap.read()

        if not ret:
            print('Video open error.')
            break

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
        img = np_img.astype(np.uint8)


        for j, boxes in enumerate(picked_boxes):
            if boxes is not None:
                for box,landmark,score in zip(boxes,picked_landmarks[j],picked_scores[j]):
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        out.write(img)
        cv2.imshow('RetinaFace-Pytorch',img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print('Now quit.')
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
