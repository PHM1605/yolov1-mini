from dataset import ShapeDataset
from torch.utils.data import DataLoader
from model import YoloV1Tiny
from loss import YoloLoss
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = ShapeDataset(2000)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = YoloV1Tiny().to(device)
criterion = YoloLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
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

torch.save(model.state_dict(), "yolov1tiny.pth")
print("Model saved to yolov1tiny.pth")
