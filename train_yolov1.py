import torch, yaml
from torch.utils.data import DataLoader
from model import Yolov1Mini
from dataset import YoloToyDataset
from losses import YoloLoss
from tqdm import tqdm 

cfg = yaml.safeload(open("config.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Yolov1Mini(cfg["grid_size"], cfg["num_boxes"], cfg["num_classes"]).to(device)
criterion = YoloLoss(cfg["grid_size"], cfg["num_boxes"], cfg["num_classes"])
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

train_ds = YoloToyDataset(cfg["data_dir"], "train", cfg["img_size"])
val_ds = YoloToyDataset(cfg["data_dir"], "val", cfg["img_size"])
train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

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