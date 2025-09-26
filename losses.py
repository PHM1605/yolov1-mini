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
  
  def forward(self, pred, target):
    loss = 0
    for box_idx in range(self.num_boxes):
      box_pred = pred[..., box_idx*5:box_idx*5+4] # (grid,grid,4)
      box_true = target[..., box_idx*5:box_idx*5+4] # (grid,grid,4)
      
  # pred_box: (grid_size,grid_size,5*num_boxes+num_classes)
  def _compute_iou(self, pred_box, true_box):
    pred_xy = pred_box[...,:2] # (grid,grid,2)
    pred_wh = pred_box[...,2:4]
    pred_tl = pred_xy - pred_wh/2 # (grid,grid,2)
    pred_br = pred_xy + pred_wh/2

