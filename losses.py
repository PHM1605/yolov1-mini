import torch 
import torch.nn as nn 



class YoloLoss(nn.Module):
  def __init__(self, grid_size=7, num_boxes=2, num_classes=2):
    super().__init__()
    self.mse = nn.MSELoss()
    self.grid_size = grid_size 
    self.num_boxes = num_boxes 
    self.num_classes = num_classes 
    self.lambda_coord = 5
    self.lambda_noobj = 0.5
    self.mse = nn.MSELoss(reduction="sum")
    self.bce = nn.BCEWithLogitsLoss(reduction="sum")
  
  # preds: (batch,grid_size,grid_size,B*5+num_classes)
  # targets: (batch,grid_size,grid_size,B*5+num_classes) (but only use 1st box)
  def forward(self, preds, targets):
    obj_mask = (targets[...,4]>0).float().unsqueeze(-1) # (batch,grid,grid,1)
    # split predictions
    box1 = preds[...,0:5] # (batch,grid,grid,4)
    box2 = preds[...,5:10] # (batch,grid,grid,4)
    cls_pred= preds[...,10:] # (batch,grid,grid,num_classes)
    # targets (using only 1st box)
    true_box = targets[...,0:5]
    cls_true = targets[...,10:]

    device = preds.device 
    grid_range = torch.arange(self.grid_size, device=device)
    cell_x = grid_range.repeat(self.grid_size,1) # repeat 1 time => each row is [0,1,..,6]
    cell_y = grid_range.repeat(self.grid_size,1).t()

    def decode(box):
      # each (batch,grid,grid,1)
      cx = (torch.sigmoid(box[...,0:1]) + cell_x) / self.grid_size # normalized, img coords
      cy = (torch.sigmoid(box[...,1:2]) + cell_y) / self.grid_size 
      w = torch.exp(box[...,2:3]) / self.grid_size 
      h = torch.exp(box[...,3:4]) / self.grid_size 
      return torch.cat([cx,cy,w,h,box[...,4:5]], dim=-1)
    
    box1_dec = decode(box1)
    box2_dec = decode(box2)

    # compute iou to assign responsible box
    iou1 = self._compute_iou(box1_dec[...,0:4], true_box[...,0:4]) # (batch,grid,grid)
    iou2 = self._compute_iou(box2_dec[...,0:4], true_box[...,0:4]) # (batch,grid,grid)
    iou_mask = (iou1>iou2).float().unsqueeze(-1) # (batch,grid,grid,1)
    # select responsible box
    pred_box = iou_mask*box1_dec + (1-iou_mask)*box2_dec # (batch,grid,grid,4)
    # box coords loss
    xy_loss = self.mse(obj_mask*pred_box[...,0:2], obj_mask*true_box[...,0:2])
    wh_loss = self.mse(
      obj_mask*torch.sqrt(pred_box[...,2:4].clamp(1e-6)),
      obj_mask*torch.sqrt(true_box[...,2:4].clamp(1e-6)))
    # objectness loss
    conf_loss_obj = self.mse(obj_mask*pred_box[...,4:5], obj_mask*true_box[...,4:5])
    # no-object loss
    noobj_mask = 1-obj_mask 
    noobj_conf_loss = (
      self.mse(noobj_mask*box1[...,4:5], torch.zeros_like(box1[...,4:5])) + 
      self.mse(noobj_mask*box2[...,4:5], torch.zeros_like(box2[...,4:5]))
    )
    # classification loss
    cls_loss = self.bce(
      obj_mask.expand_as(cls_pred) * cls_pred,
      obj_mask.expand_as(cls_true) * cls_true 
    )
    # total loss 
    total = (
      self.lambda_coord*(xy_loss+wh_loss) + 
      conf_loss_obj + 
      self.lambda_noobj * noobj_conf_loss +
      cls_loss
    )
    return total 
      
  # pred_box: (grid_size,grid_size,5)
  def _compute_iou(self, pred_box, true_box):
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

    pred_area = (pred_br[...,0]-pred_tl[...,0])*(pred_br[...,1]-pred_tl[...,1])
    true_area = (true_br[...,0]-true_tl[...,0])*(true_br[...,1]-true_tl[...,1])
    union_area = pred_area + true_area - inter_area 
    return inter_area / (union_area+1e-6) # (grid,grid)
