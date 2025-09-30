import torch.nn.functional as F 
import torch.nn as nn
import torch

# box1, box2: (N,4)
def box_iou_xywh(box1, box2):
  # xywh => x1y1x2y2
  b1_x1 = box1[:,0] - box1[:,2]/2
  b1_y1 = box1[:,1] - box1[:,3]/2
  b1_x2 = box1[:,0] + box1[:,2]/2
  b1_y2 = box1[:,1] + box1[:,3]/2

  b2_x1 = box2[:,0] - box2[:,2]/2
  b2_y1 = box2[:,1] - box2[:,3]/2
  b2_x2 = box2[:,0] + box2[:,2]/2
  b2_y2 = box2[:,1] + box2[:,3]/2

  inter_x1 = torch.max(b1_x1, b2_x1)
  inter_y1 = torch.max(b1_y1, b2_y1)
  inter_x2 = torch.min(b1_x2, b2_x2)
  inter_y2 = torch.min(b1_y2, b2_y2)

  inter_area = (inter_x2-inter_x1).clamp(0) * (inter_y2-inter_y1).clamp(0)
  b1_area = (b1_x2-b1_x1) * (b1_y2-b1_y1)
  b2_area = (b2_x2-b2_x1) * (b2_y2-b2_y1)
  return inter_area / (b1_area+b2_area-inter_area+1e-6)

class YoloLoss(nn.Module):
  def __init__(self, S=7, B=2, C=2, lambda_coord=5, lambda_noobj=0.5):
    super().__init__()
    self.S, self.B, self.C = S, B, C 
    self.lambda_coord = lambda_coord
    self.lambda_noobj = lambda_noobj
  
  # pred, target: (N,S,S,B*5+C)
  def forward(self, pred, target):
    N = pred.size(0)
    obj_mask = target[...,4] > 0 # (N,S,S)
    noobj_mask = target[...,4] == 0

    pred_boxes = pred[...,:self.B*5].view(N,self.S,self.S,self.B,5) # (N,S,S,B,5)
    target_boxes = target[...,:self.B*5].view(N,self.S,self.S,self.B,5) # (N,S,S,B,5)
    gt_box = target_boxes[...,0,:4] # (N,S,S,4) 
    # compute iou for each predicted box vs gt
    ious = []
    for b in range(self.B):
      ious.append(box_iou_xywh(
        pred_boxes[...,b,:4][obj_mask], # (N,S,S,4)=>(num_boxes,4)
        gt_box[obj_mask] # (num_boxes,4)
      ))
    ious = torch.stack(ious, dim=1) # (num_boxes,B); B is template-box-number
    best_box = ious.argmax(dim=1) # (num_boxes,)
    # coord loss 
    loss_xy, loss_wh, loss_obj = 0, 0, 0
    idx = 0
    for b in range(self.B):
      mask_b = obj_mask.clone() # (N,S,S)
      mask_b[obj_mask] = (best_box==b) # most of mask_b is False, True only where object is and best_box
      # coord loss
      pred_xy = pred_boxes[...,b,0:2][mask_b] # (num_boxes_with_b_bestbox,2)
      target_xy = gt_box[mask_b][...,0:2] # (num_boxes_with_b_bestbox,2)
      loss_xy += F.mse_loss(pred_xy, target_xy, reduction="sum")

      pred_wh = torch.sqrt(pred_boxes[...,b,2:4].clamp(1e-6))[mask_b]
      target_wh = torch.sqrt(gt_box[mask_b][...,2:4].clamp(1e-6))
      loss_wh += F.mse_loss(pred_wh, target_wh, reduction="sum")
      # obj loss 
      pred_conf = pred_boxes[...,b,4][mask_b]
      target_conf = ious[:,b][best_box==b]
      loss_obj += F.mse_loss(pred_conf, target_conf, reduction="sum")

    # no-obj loss 
    loss_noobj = 0
    for b in range(self.B):
      loss_noobj += F.mse_loss(
        pred_boxes[...,b,4][noobj_mask],
        target_boxes[...,b,4][noobj_mask],
        reduction = "sum"
      )
    # class loss
    class_pred = pred[obj_mask][...,5*self.B:]
    class_target = target[obj_mask][...,5*self.B:]
    loss_cls = F.mse_loss(class_pred, class_target, reduction="sum")

    total = self.lambda_coord*(loss_xy+loss_wh) + loss_obj + self.lambda_noobj*loss_noobj + loss_cls
    return total / N
