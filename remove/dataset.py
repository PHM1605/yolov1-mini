import glob, os, torch, math
from torch.utils.data import Dataset 
from PIL import Image 
from torchvision import transforms

class YoloToyDataset(Dataset):
  def __init__(self, data_dir, split, img_size):
    self.img_dir = os.path.join(data_dir, 'images', split)
    self.img_size = img_size 
    self.grid_size = 7
    self.num_classes = 2
  
    self.img_paths = sorted(glob.glob(f"{self.img_dir}/*.jpg"))
    self.transform = transforms.Compose([
      transforms.Resize((img_size, img_size)),
      transforms.ToTensor()
    ])
  
  def __len__(self):
    return len(self.img_paths)
  
  def __getitem__(self, idx):
    img_path = self.img_paths[idx]
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    img = Image.open(img_path).convert("RGB")
    img = self.transform(img)
    target = torch.zeros((self.grid_size, self.grid_size, 5+self.num_classes)) # target = only 1 box
    
    if os.path.exists(label_path):
      with open(label_path) as f:
        rows = f.readlines()

    for row in rows:
      row = row.strip().split()
      cl, cx, cy, box_w, box_h = map(float, row)
      cl = int(cl)
      # i, j: cell row and cell column
      i = int(cy*self.grid_size)
      j = int(cx*self.grid_size)
      # convert to cell-relative coords
      cell_x = cx*self.grid_size-j
      cell_y = cy*self.grid_size-i
      # width/height of box in [cell] unit
      width_cell = box_w * self.grid_size 
      height_cell = box_h * self.grid_size 

      # 1 cell has only 1 object
      if target[i,j,5] == 0:
        target[i,j,5] = 1
        box = torch.tensor([cell_x, cell_y, width_cell, height_cell]) # (x-within-cell,y-within-cell,w,h)
        target[i,j,0:4] = box 
        target[i,j,5+cl] = 1
    return img, target
  