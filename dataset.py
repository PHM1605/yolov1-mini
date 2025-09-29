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
      # transforms.Resize((img_size, img_size)),
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

        col = int(cx*self.grid_size)
        row = int(cy*self.grid_size)
        # convert to cell-relative coords
        cell_x = cx*self.grid_size-col
        cell_y = cy*self.grid_size-row
        box = torch.tensor([cell_x, cell_y, math.sqrt(box_w), math.sqrt(box_h), 1.0]) # (cx,cy,w,h,conf)
        target[row,col,0:5] = box 
        target[row,col,5+int(cl)] = 1
        # print(f"Image {idx}: cell=({j},{i}), box={box.tolist()}, class={int(cl)}")
    return img, target
  