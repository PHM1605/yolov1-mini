from dataset import YoloToyDataset
from torch.utils.data import DataLoader
from model import YoloV1Tiny
from loss import YoloLoss
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def init_weights(m):
  if isinstance(m, torch.nn.Conv2d):
    torch.nn.init.kaiming_normal_(m.weight, a=0.1)
    if m.bias is not None:
      torch.nn.init.constant_(m.bias, 0.0)
  elif isinstance(m, torch.nn.Linear):
    torch.nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
      torch.nn.init.constant_(m.bias, 0.0)


dataset = YoloToyDataset("train")
loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = YoloV1Tiny().to(device)
model.apply(init_weights)
criterion = YoloLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
  total = 0
  for imgs, targets in loader:
    imgs = imgs.to(device)
    targets = targets.to(device)
    preds = model(imgs)
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total += loss.item()
  print(f"Epoch {epoch+1} Loss: {total/len(loader):.4f}")

torch.save(model.state_dict(), "yolov1tiny.pt")
print("Model saved to yolov1tiny.pth")
