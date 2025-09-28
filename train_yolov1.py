import torch, yaml
from torch.utils.data import DataLoader
from model import Yolov1Mini
from dataset import YoloToyDataset
from losses import YoloLoss
from tqdm import tqdm 

cfg = yaml.safe_load(open("config.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Yolov1Mini(cfg["grid_size"], cfg["num_boxes"], cfg["num_classes"]).to(device)
criterion = YoloLoss(cfg["grid_size"], cfg["num_boxes"], cfg["num_classes"])
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

train_ds = YoloToyDataset(cfg["data_dir"], "train", cfg["img_size"])
val_ds = YoloToyDataset(cfg["data_dir"], "val", cfg["img_size"])
train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

# box: (5,) = [x,y,w,h,conf]
def decode(box, row, col, grid_size):
  cx = (box[0]+col) / grid_size 
  cy = (box[1]+row) / grid_size
  return torch.tensor([cx,cy,box[2],box[3],box[4]]) 

for epoch in range(cfg["epochs"]):
  model.train()
  total_loss = 0
  for img, target in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
    img, target = img.to(device), target.to(device)
    pred = model(img)
    loss = criterion(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
  # Validation
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for img, target in val_loader:
      img = img.to(device)
      pred = model(img) # (batch,grid,grid,5+num_classes)
      pred = pred.cpu()
      target = target.cpu()
      batch, grid_size, grid_size, _ = pred.shape 
      for b in range(batch):
        # find cell with highest confidence
        cell_conf = pred[b, :, :, 4] # (grid,grid)
        max_idx = cell_conf.argmax() # e.g. max at cell (3,5) => max_idx=3*7+5=26
        i, j = divmod(max_idx.item(), grid_size) # divmod(26,7)=>(26//7,26%7)=>(3,5)

        pred_box_raw = pred[b,i,j,0:5] # (5,)
        pred_box = decode(pred_box_raw, i, j, grid_size)
        pred_cls = pred[b,i,j,5:].argmax().item()
        true_box = target[b,i,j,0:5]
        true_cls = target[b,i,j,5:].argmax().item()
        # only consider cells that have object
        if target[b,i,j,4] == 1:
          print("WE HERE")
          iou = criterion._compute_iou(pred_box[None,None,:4], true_box[None,None,:4])[0,0]
          print("IOU: ", iou)
          if pred_cls == true_cls and iou>0.5:
            correct += 1
          total += 1
  acc = correct / max(total, 1)
  print(f"Val accuracy: {acc:.4f}")
      