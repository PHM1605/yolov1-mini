import glob, os, torch 
from torch.utils.data import Dataset 
from PIL import Image 
from torchvision import transforms

class YoloToyDataset(Dataset):
  def __init__(self, data_dir, split, img_size):
    self.img_dir = os.path.join(data_dir, 'images', split)
    self.img_size = img_size 
    self.grid_size = 7
    self.num_boxes = 2
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
    target = torch.zeros((self.grid_size, self.grid_size, self.num_boxes*5+self.num_classes))
    
    if os.path.exists(label_path):
      with open(label_path) as f:
        rows = f.readlines()
      for row in rows:
        row = row.strip().split()
        cl, cx, cy, box_w, box_h = map(float, row)

        i = int(cx*self.grid_size)
        j = int(cy*self.grid_size)
        # convert to cell-relative coords
        cell_x = cx*self.grid_size-i 
        cell_y = cy*self.grid_size-j
        box = torch.tensor([cell_x, cell_y, box_w, box_h, 1.0]) # (cx,cy,w,h,conf)
        # fill first box (ASSUME 1 object per cell only)
        if target[j,i,4] == 0:
          target[j,i,4] = 1
          target[j,i,0:5] = box 
          target[j,i,self.num_boxes*5+int(cl)] = 1
        # print(f"Image {idx}: cell=({j},{i}), box={box.tolist()}, class={int(cl)}")
    return img, target
  