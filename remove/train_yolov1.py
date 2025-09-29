import torch, yaml
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import YoloToyDataset
from losses import YoloLoss, decode_pred, decode_target
from tqdm import tqdm 

cfg = yaml.safe_load(open("config.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_fn(train_loader, model, optimizer, loss_fn):
  loop = tqdm(train_loader, leave=True)
  mean_loss = []
  for batch_idx, (x,y) in enumerate(loop):
    x, y = x.to(device), y.to(device)
    out = model(x)
    loss = loss_fn(out, y)
    mean_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loop.set_postfix(loss=loss.item())
  print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")

def main():
  model = Yolov1(grid_size=7, num_boxes=2, num_classes=3).to(device)
  optimizer = optim.Adam(
    model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
  )
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3)
  loss_fn = YoloLoss()

  train_ds = YoloToyDataset(cfg["data_dir"], "train", cfg["img_size"])
  val_ds = YoloToyDataset(cfg["data_dir"], "val", cfg["img_size"])
  train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

  for epoch in range(cfg["epochs"]):
    train_fn(train_loader, model, optimizer, loss_fn)
  #   pred_boxes, target_boxes = get_bboxes(
  #     train_loader, model, iou_threshold=0.5, threshold=0.4
  #   )
  #   mean_avg_prec = mean_average_precision(
  #     pred_boxes, target_boes, iou_threshold=0.5, box_format="midpoint"
  #   )
  #   print(f"Train mAP: {mean_avg_prec}")
  #   scheduler.step(mean_avg_prec)
  # # save checkpoint 
  # checkpoint = {
  #   "state_dict": model.state_dict(),
  #   "optimizer": optimizer.state_dict()
  # }
  # save_checkpoint(checkpoint, filename="model.pth")

main()

# for epoch in range(cfg["epochs"]):
#   model.train()
#   total_loss = 0
#   for img, target in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
#     img, target = img.to(device), target.to(device)
#     pred = model(img)
#     loss = criterion(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     scheduler.step()
#     total_loss += loss.item()
#   print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
#   # Validation
#   model.eval()
#   correct = 0
#   total = 0
#   with torch.no_grad():
#     for img, target in val_loader:
#       img = img.to(device)
#       pred = model(img) # (batch,grid,grid,5+num_classes)
#       pred = pred.cpu()
#       target = target.cpu()
#       batch, grid_size, grid_size, _ = pred.shape 
#       for b in range(batch):
#         # find cell with highest confidence
#         cell_conf = pred[b, :, :, 4] # (grid,grid)
#         max_idx = cell_conf.argmax() # e.g. max at cell (3,5) => max_idx=3*7+5=26
#         row, col = divmod(max_idx.item(), grid_size) # divmod(26,7)=>(26//7,26%7)=>(3,5)

#         pred_box_raw = pred[b,row,col,0:5] # (5,)
#         pred_box = decode_pred(
#           pred_box_raw.unsqueeze(0).unsqueeze(0), 
#           torch.tensor([[col]]), 
#           torch.tensor([[row]]), 
#           grid_size)[0,0]
#         pred_cls = pred[b,row,col,5:].argmax().item()

#         true_box_raw = target[b,row,col,0:5]
#         true_box = decode_target(
#           true_box_raw.unsqueeze(0).unsqueeze(0),
#           torch.tensor([[col]]),
#           torch.tensor([[row]]),
#           grid_size)[0,0]
#         true_cls = target[b,row,col,5:].argmax().item()
#         # only consider cells that have object
#         if target[b,row,col,4] == 1:
#           print("WE HERE")
#           print(pred_box)
#           print(true_box)
#           iou = criterion._compute_iou(pred_box[None,None,:4], true_box[None,None,:4])[0,0]
#           print("IOU: ", iou)
#           if pred_cls == true_cls and iou>0.5:
#             correct += 1
#           total += 1
#   acc = correct / max(total, 1)
#   print(f"Val accuracy: {acc:.4f}")
      