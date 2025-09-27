import torch 
import torch.nn as nn

class Yolov1Mini(nn.Module):
  def __init__(self, num_cells=7, num_boxes=2, num_classes=2):
    super().__init__()
    self.num_cells = num_cells
    self.num_boxes = num_boxes
    self.num_classes = num_classes 
    self.backbone = nn.Sequential(
      # (channel_in, channel_out, kernel, stride, padding)
      nn.Conv2d(3, 64, 7, 2, 3), nn.LeakyReLU(0.1), nn.MaxPool2d(kernel_size=2, stride=2), # (16,112,112)
      nn.Conv2d(64, 192, 3, 1, 1), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2), # (32,56,56)
      nn.Conv2d(192, 128, 1, 1, 0), nn.LeakyReLU(), nn.MaxPool2d(2), # (64,28,28)
      # nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), # (128,14,14)
      # nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), # (256,7,7)
      # nn.Flatten()
    )
    # self.fc = nn.Sequential(
    #   nn.Linear(256*(self.num_cells*self.num_cells), 512),
    #   nn.ReLU(),
    #   nn.Linear(512, self.num_cells*self.num_cells*(self.num_boxes*5+self.num_classes))
    # )

  def forward(self, x):
    x = self.backbone(x)
    # x = self.fc(x)
    # return x.view(-1, self.num_cells, self.num_cells, self.num_boxes*5+self.num_classes)
    return x 
