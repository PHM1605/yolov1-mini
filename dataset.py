import glob, os, torch 
from torch.utils.data import Dataset 
from PIL import Image 
from torchvision import transforms

class YoloToyDataset(Dataset):
  def __init__(self, data_dir, split, img_size):
    self.img_dir = os.path.join(data_dir, split)
    self.img_size = img_size 
    self.grid_size = 7
    self.num_boxes = 2
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
    label_path = img_path.replace(".jpg", ".txt")
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    target = torch.zeros((self.grid_size, self.grid_size, self.num_boxes*5+self.num_classes))
    
    if os.path.exists(label_path):
      with open(label_path) as f:
        for line in f:
          cl, x, y, w, h = map(float, line.strip().split())
          i = int(x*self.grid_size)
          j = int(y*self.grid_size)
          box = torch.tensor([x, y, w, h, 1.0]) # (cx,cy,w,h,conf)
          # fill first box (ASSUME 1 object per cell only)
          if target[j,i,4] == 0:
            target[j,i,0:5] = box 
            target[j,i,10+int(cl)] = 1
    return img, target
  