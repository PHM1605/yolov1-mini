import cv2, torch
import numpy as np 
from model import Yolov1Mini
from torchvision import transforms

transform = transforms.Compose([
  transforms.ToTensor()
])

img = cv2.imread("data/images/train/00009.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = transform(img).unsqueeze(0) # (1,3,H,W)
# with open("data/labels/train/00009.txt", "r") as f:
#   rows = f.readlines()
# row = [r.strip().split() for r in rows][0]
# cl, cx, cy, box_w, box_h = map(float, row)
# img_h, img_w, _ = img.shape

# # convert to pixels
# x1 = int((cx-box_w/2)*img_w) 
# y1 = int((cy-box_h/2)*img_h)
# x2 = int(x1 + box_w*img_w) 
# y2 = int(y1 + box_h*img_h)

# color = (0,255,0) if int(cl)==0 else (0,0,255)
# cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)
# cv2.imwrite('test.png', img)

yolo_test = Yolov1Mini()
res = yolo_test(img)
print(res.shape)