import glob, os, torch, math
from torch.utils.data import Dataset 
from PIL import Image 
from torchvision import transforms

class YoloToyDataset(Dataset):
  def __init__(self, split, data_dir="data", img_size=448):
    self.img_dir = os.path.join(data_dir, 'images', split)
    self.img_size = img_size 
    self.grid_size = 7
    self.classes = ["triangle", "circle"]
    self.num_classes = len(self.classes)
  
    self.img_paths = sorted(glob.glob(f"{self.img_dir}/*.jpg"))
    self.transform = transforms.Compose([
      transforms.Resize((img_size, img_size)),
      transforms.ToTensor()
    ])
  
  def __len__(self):
    return len(self.img_paths)
  
  def __getitem__(self, idx):
    img_path = self.img_paths[idx]
    # print(img_path)
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    img = Image.open(img_path).convert("RGB")
    img = self.transform(img) # (3,H,W) with values [0..1]
    target = torch.zeros((self.grid_size, self.grid_size, 5*2+self.num_classes)) # target = 2 boxes
    
    if os.path.exists(label_path):
      with open(label_path) as f:
        rows = f.readlines()

    for row in rows:
      row = row.strip().split()
      cl, cx, cy, w_norm, h_norm = map(float, row)
      cl = int(cl)
      # center in [cell] unit
      gx, gy = cx*self.grid_size, cy*self.grid_size
      # gi, gj: center in which cell with gj=row, gi=col 
      gi, gj = int(gx), int(gy) 
      # convert to cell-relative coords
      x_cell, y_cell = gx-gi, gy-gj

      for b in range(2):
        target[gj,gi,b*5:(b+1)*5] = torch.tensor([x_cell,y_cell,w_norm,h_norm,1.0])
      target[gj,gi,10+cl] = 1.0

    return img, target
  