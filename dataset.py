import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw 
import random 
import numpy as np 

class ShapeDataset(Dataset):
  def __init__(self, n=2000, grid_size=7, img_size=448):
    self.n = n
    self.grid_size = grid_size
    self.img_size = img_size 
    self.classes = ["circle", "square"]
  
  def __len__(self): 
    return self.n 
  
  def __getitem__(self, idx):
    img = Image.new("RGB", (self.img_size, self.img_size), "black")
    draw = ImageDraw.Draw(img)

    # random shape parameters
    cl = random.randint(0, 1)
    cx, cy = random.randint(100,348), random.randint(100,348)
    size = random.randint(40,100)
    x1, y1, x2, y2 = cx-size, cy-size, cx+size, cy+size 

    # circle
    if cl == 0:
      draw.ellipse([x1,y1,x2,y2], outline="white", fill="white")
    # square
    else:
      draw.rectangle([x1,y1,x2,y2], outline="white", fill="white")
    
    target = torch.zeros((self.grid_size, self.grid_size, 5*2+len(self.classes)))
    cell_size = self.img_size / self.grid_size
    gx, gy = cx/cell_size, cy/cell_size # center in [cell] unit
    gi, gj = int(gx), int(gy) # center in which cell 

    x_cell, y_cell = gx-gi, gy-gj 
    w_norm, h_norm = (2*size)/self.img_size, (2*size)/self.img_size 

    target[gj,gi,0:5] = torch.tensor([x_cell, y_cell, w_norm, h_norm, 1.0])
    target[gj,gi,10+cl] = 1.0

    # img: (3,H,W)
    # target: (grid,grid,5*2+num_classes)
    return torch.tensor(np.array(img)).permute(2,0,1).float()/255.0, target 