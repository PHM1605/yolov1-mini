import torch 
import torch.nn as nn 

# convert box in cell-coords-xy and norm-coords-sqrt-wh to image-coords
# box: (batch,grid,grid,4)
def decode_pred(box, cell_x, cell_y, grid_size):
  cx = (torch.sigmoid(box[...,0:1]) + cell_x) / grid_size
  cy = (torch.sigmoid(box[...,1:2]) + cell_y) / grid_size
  w = torch.sigmoid(box[..., 2:3])**2
  h = torch.sigmoid(box[..., 3:4])**2
  conf = torch.sigmoid(box[...,4:5])
  return torch.cat([cx,cy,w,h,conf], dim=-1)

def decode_target(box, cell_x, cell_y, grid_size):
  cx = (box[...,0:1]+cell_x) / grid_size
  cy = (box[...,1:2]+cell_y) / grid_size 
  w = box[...,2:3]**2
  h = box[...,3:4]**2
  return torch.cat([cx, cy, w, h, box[...,4:5]], dim=-1)

# pred_box: (batch,grid_size,grid_size,4) or (num_boxes,4)
def intersection_over_union(pred_box, true_box):
  pred_xy = pred_box[...,:2] # (grid,grid,2)
  pred_wh = pred_box[...,2:4]
  pred_tl = pred_xy - pred_wh/2 # (grid,grid,2)
  pred_br = pred_xy + pred_wh/2

  true_xy = true_box[...,:2]
  true_wh = true_box[...,2:4]
  true_tl = true_xy - true_wh/2
  true_br = true_xy + true_wh/2

  tl = torch.max(pred_tl, true_tl) # (grid,grid,2)
  br = torch.min(pred_br, true_br) # (grid,grid,2)
  inter = (br-tl).clamp(min=0) # (grid,grid,2)
  inter_area = inter[...,0]*inter[...,1] # (grid,grid)

  pred_area = abs(pred_br[...,0]-pred_tl[...,0])*(pred_br[...,1]-pred_tl[...,1])
  true_area = abs(true_br[...,0]-true_tl[...,0])*(true_br[...,1]-true_tl[...,1])
  union_area = pred_area + true_area - inter_area 
  return inter_area / (union_area+1e-6) # (grid,grid)

class YoloLoss(nn.Module):
  def __init__(self, grid_size=7, num_boxes=2, num_classes=2):
    super().__init__()
    self.grid_size = grid_size 
    self.num_boxes = num_boxes 
    self.num_classes = num_classes 
    self.lambda_coord = 5 # weight for coordinates loss
    self.lambda_noobj = 0.5 # weight for no-object loss 
    self.sse = nn.MSELoss(reduction="sum")
  
  # preds: (batch,grid_size,grid_size,B*5+num_classes)
  # targets: (batch,grid_size,grid_size,5+num_classes)
  def forward(self, preds, targets):
    preds = preds.reshape(-1, self.grid_size, self.grid_size, self.num_classes+self.num_boxes*5)
    # compute iou to assign responsible box
    iou1 = intersection_over_union(preds[...,:4], targets[...,:4]) # (batch,grid,grid)
    iou2 = intersection_over_union(preds[...,4:8], targets[...,:4]) # (batch,grid,grid)



    obj_mask = (targets[...,4]>0).float().unsqueeze(-1) # (batch,grid,grid,1)
    noobj_mask = 1 - obj_mask
    # split predictions
    box1 = preds[...,0:5] # (batch,grid,grid,4)
    box2 = preds[...,5:10] # (batch,grid,grid,4)
    cls_pred= preds[...,10:] # (batch,grid,grid,num_classes)
    # targets 
    true_box = targets[...,0:5]
    cls_true = targets[...,5:]

    device = preds.device 
    grid_range = torch.arange(self.grid_size, device=device)
    cell_x = grid_range.repeat(self.grid_size,1) # repeat 7 time along row, 1 time along column
    cell_y = grid_range.repeat(self.grid_size,1).t()
    # decode predictions
    box1_dec = decode_pred(box1, cell_x, cell_y, self.grid_size)
    box2_dec = decode_pred(box2, cell_x, cell_y, self.grid_size)
    # decode true target
    true_dec = decode_target(true_box, cell_x, cell_y, self.grid_size)
    # compute iou to assign responsible box
    iou1 = self._compute_iou(box1_dec[...,:4], true_dec[...,:4]) # (batch,grid,grid)
    iou2 = self._compute_iou(box2_dec[...,:4], true_dec[...,:4]) # (batch,grid,grid)
    iou_mask = (iou1>iou2).float().unsqueeze(-1) # (batch,grid,grid,1)
    # select responsible box
    pred_box = iou_mask*box1_dec + (1-iou_mask)*box2_dec # (batch,grid,grid,4)
    iou_responsible = (iou_mask*iou1.unsqueeze(-1)+(1-iou_mask)*iou2.unsqueeze(-1))
    # box coords loss
    xy_loss = self.sse(obj_mask*pred_box[...,0:2], obj_mask*true_dec[...,0:2])
    wh_loss = self.sse(obj_mask*torch.sqrt(pred_box[...,2:4].clamp(1e-6)), obj_mask*torch.sqrt(true_dec[...,2:4]))
    # objectness loss
    conf_loss_obj = self.sse(obj_mask*pred_box[...,4:5], obj_mask*iou_responsible)
    # no-object loss
    noobj_conf_loss = self.sse(noobj_mask*pred_box[...,4:5], torch.zeros_like(pred_box[...,4:5]))
    # classification loss
    cls_loss = self.sse(obj_mask*torch.softmax(cls_pred,dim=-1), obj_mask*cls_true)
    # total loss 
    total = (
      self.lambda_coord*(xy_loss+wh_loss) + 
      conf_loss_obj + 
      self.lambda_noobj * noobj_conf_loss +
      cls_loss
    )
    return total 
