from PIL import Image, ImageDraw
from dataset import ShapeDataset
from model import YoloV1Tiny
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# boxes: (N,4)
# scores: (N,)
def nms(boxes, scores, iou_thresh=0.5):
  if len(boxes) == 0:
    return []
  boxes = torch.tensor(boxes, dtype=torch.float32)
  scores = torch.tensor(scores, dtype=torch.float32)
  x1, y1, x2, y2 = boxes.T # (N,) each
  areas = (x2-x1).clamp(0)*(y2-y1).clamp(0)
  order = scores.argsort(descending=True)

  keep = [] # which box-index we want to keep
  while order.numel() > 0:
    i = order[0].item()
    keep.append(i)
    if order.numel() == 1:
      break 
    
    xx1 = torch.max(x1[i], x1[order[1:]])
    yy1 = torch.max(y1[i], y1[order[1:]])
    xx2 = torch.min(x2[i], x2[order[1:]])
    yy2 = torch.min(y2[i], y2[order[1:]])
    inter = (xx2-xx1).clamp(0) * (yy2-yy1).clamp(0)
    # iou between the max-box and the rest
    iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
    # find where NOT overlap with max-box
    inds = (iou<=iou_thresh).nonzero(as_tuple=False).squeeze()
    if inds.numel() == 0:
      break 
    # order=[5,3,1,2,6,0]; sorting of scores in descending order
    # inds = [0,2,3] (index of the rest boxes [3,1,2,6,0])
    # inds+1 = [1,3,4] => order becomes [3,2,6]
    order = order[inds+1]
  
  return keep


# img: (3,H,W)
# pred: (S,S,B*5+C)
def save_prediction(img, pred, filename="result.png", S=7, img_size=448, conf_thresh=0.2, iou_thresh=0.5, class_names=None):
  img = (img.permute(1,2,0).cpu().numpy()*255).astype("uint8") # (H,W,3)
  img_pil = Image.fromarray(img)
  draw = ImageDraw.Draw(img_pil)
  boxes, labels = decode_prediction(pred, S=S, img_size=img_size, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
  # print(boxes)
  # print(labels)
  for (x1, y1, x2, y2), cl in zip(boxes, labels):
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    color = "red" if cl == 0 else "blue"
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    label_text = str(cl) if class_names is None else class_names[cl]
    draw.text((x1, max(0, y1-10)), label_text, fill=color)
  img_pil.save(filename)
  print(f"Saved prediction result to {filename}")

def decode_prediction(pred, S=7, B=2, img_size=448, conf_thresh=0.2, iou_thresh=0.5):
  boxes, labels, scores = [], [], []
  cell_size = img_size / S 
  for i in range(S):
    for j in range(S):
      # scan the 2 boxes predicting this cell
      for b in range(B):
        p = pred[j,i,b*5:(b+1)*5] # (x,y,w,h,conf)
        conf = p[4].item()
        if conf < conf_thresh:
          continue
        # decode box
        x, y, w, h = p[0:4]
        cx = (i+x.item()) * cell_size 
        cy = (j+y.item()) * cell_size 
        w, h = w.item()*img_size, h.item()*img_size
        # pick class with highest score
        class_scores = pred[j,i,B*5:]
        cl = torch.argmax(class_scores).item()
        boxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
        labels.append(cl)
        scores.append(conf)
  # apply nms 
  keep = nms(boxes, scores, iou_thresh=iou_thresh)
  boxes = [boxes[i] for i in keep]
  labels = [labels[i] for i in keep]

  return boxes, labels 

dataset = ShapeDataset(2000)
test_img, _ = dataset[100]
model = YoloV1Tiny().to(device)
model.load_state_dict(torch.load("yolov1tiny.pth", map_location=device))
model.eval()
with torch.no_grad():
  pred = model(test_img.unsqueeze(0).to(device))[0].cpu()
save_prediction(test_img, pred, filename="predicted.png", class_names=["circle","square"], conf_thresh=0.2)