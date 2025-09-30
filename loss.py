import torch.nn.functional as F 
import torch.nn as nn
import torch

class YoloLoss(nn.Module):
  def __init__(self, S=7, B=2, C=2, lambda_coord=5, lambda_noobj=0.5):
    super().__init__()
    self.S, self.B, self.C = S, B, C 
    self.lambda_coord = lambda_coord
    self.lambda_noobj = lambda_noobj
  
  def forward(self, pred, target):
    obj_mask = target[...,4] > 0 
    noobj_mask = target[...,4] == 0

    # coord loss 
    box_pred = pred[obj_mask][...,0:4]
    box_target = target[obj_mask][...,0:4]
    loss_xy = F.mse_loss(box_pred[...,0:2], box_target[...,0:2], reduction="sum")
    loss_wh = F.mse_loss(
      torch.sqrt(box_pred[...,2:4].clamp(1e-6)),
      torch.sqrt(box_target[...,2:4].clamp(1e-6)),
      reduction="sum"
    )

    # obj, nobj confidence loss
    conf_pred = pred[...,4]
    conf_target = target[...,4]
    loss_obj = F.mse_loss(conf_pred[obj_mask], conf_target[obj_mask], reduction="sum")
    loss_noobj = F.mse_loss(conf_pred[noobj_mask], conf_target[noobj_mask], reduction="sum")

    # class loss
    class_pred = pred[obj_mask][...,5*self.B:]
    class_target = target[obj_mask][...,5*self.B:]
    loss_cls = F.mse_loss(class_pred, class_target, reduction="sum")

    total = self.lambda_coord*(loss_xy+loss_wh) + loss_obj + self.lambda_noobj*loss_noobj + loss_cls
    return total / pred.size(0)
