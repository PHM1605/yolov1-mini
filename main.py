from dataset import ShapeDataset
from torch.utils.data import DataLoader
from model import YoloV1Tiny
from loss import YoloLoss
import torch
from PIL import Image, ImageDraw

def decode_prediction(pred, S=7, B=2, img_size=448):
  boxes, labels = [], []
  cell_size = img_size / S 
  for i in range(S):
    for j in range(S):
      p = pred[j,i]
      conf = p[4]
      if conf > 0.5:
        x, y, w, h = p[0:4]
        cx = (i+x.item()) * cell_size 
        cy = (j+y.item()) * cell_size 
        w, h = w.item()*img_size, h.item()*img_size 
        cl = torch.argmax(p[5*B:]).item()
        boxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
        labels.append(cl)
  return boxes, labels 

# img: (3,H,W)
def save_prediction(img, pred, filename="result.png", S=7, img_size=448, conf_thresh=0.5):
  img = (img.permute(1,2,0).cpu().numpy()*255).astype("uint8") # (H,W,3)
  img_pil = Image.fromarray(img)
  draw = ImageDraw.Draw(img_pil)
  boxes, labels = decode_prediction(pred, S=S, img_size=img_size)
  for (x1, y1, x2, y2), cl in zip(boxes, labels):
    color = "red" if cl == 0 else "blue"
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    draw.text((x1, y1-10), f"{cl}", fill=color)
  img_pil.save(filename)
  print(f"Saved prediction result to {filename}")

dataset = ShapeDataset(2000)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = YoloV1Tiny()
criterion = YoloLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
  total = 0
  for imgs, targets in loader:
    preds = model(imgs)
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total += loss.item()
  print(f"Epoch {epoch+1} Loss: {total/len(loader):.4f}")

test_img, _ = dataset[0]
model.eval()
with torch.no_grad():
  pred = model(test_img.unsqueeze(0))[0]
save_prediction(test_img, pred, filename="predicted.png")