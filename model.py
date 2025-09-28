import torch 
import torch.nn as nn

def conv_block(c_in, c_out, k, s, p, use_pool):
  layers = []
  # (channel_in, channel_out, kernel, stride, padding)
  layers.append(nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False))
  layers.append(nn.LeakyReLU())
  if use_pool:
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  
  return nn.Sequential(*layers)

class Yolov1Mini(nn.Module):
  def __init__(self, grid_size=7, num_boxes=2, num_classes=2):
    super().__init__()
    self.grid_size = grid_size
    self.num_boxes = num_boxes
    self.num_classes = num_classes 
    
    backbone_layers = []
    backbone_layers.append(conv_block(3, 64, k=7, s=2, p=3, use_pool=True)) # (batch,64,56,56)
    backbone_layers.append(conv_block(64, 192, k=3, s=1, p=1, use_pool=True)) # (batch,192,28,28)
    backbone_layers.append(conv_block(192, 128, k=1, s=1, p=0, use_pool=False)) # (batch,128,28,28)
    backbone_layers.append(conv_block(128, 256, k=3, s=1, p=1, use_pool=False))
    backbone_layers.append(conv_block(256, 256, k=1, s=1, p=0, use_pool=False))
    backbone_layers.append(conv_block(256, 512, k=3, s=1, p=1, use_pool=True)) # (batch,512,14,14)
    for i in range(4):
      backbone_layers.append(conv_block(512, 256, k=1, s=1, p=0, use_pool=False))
      backbone_layers.append(conv_block(256, 512, k=3, s=1, p=1, use_pool=False))
    backbone_layers.append(conv_block(512, 512, k=1, s=1, p=0, use_pool=False))
    backbone_layers.append(conv_block(512, 1024, k=3, s=1, p=1, use_pool=True))
    for i in range(2):
      backbone_layers.append(conv_block(1024, 512, k=1, s=1, p=0, use_pool=False))
      backbone_layers.append(conv_block(512, 1024, k=3, s=1, p=1, use_pool=False))
    backbone_layers.append(conv_block(1024, 1024, k=3, s=1, p=1, use_pool=False))
    # backbone_layers.append(conv_block(1024, 1024, k=3, s=2, p=1, use_pool=False)) # remove the stride=2 to make output 7x7x30
    backbone_layers.append(conv_block(1024, 1024, k=3, s=1, p=1, use_pool=False))
    backbone_layers.append(conv_block(1024, 1024, k=3, s=1, p=1, use_pool=False)) # (batch,1024,7,7)

    # fully-connected layers
    fc_layers = []
    fc_layers.append(nn.Flatten()) # (4096,)
    fc_layers.append(nn.Linear(1024*grid_size*grid_size, 4096))
    fc_layers.append(nn.LeakyReLU(0.1))
    fc_layers.append(nn.Dropout(0.5))
    fc_layers.append(nn.Linear(4096, grid_size*grid_size*(num_boxes*5+num_classes)))

    self.backbone = nn.Sequential(*backbone_layers) 
    self.fc = nn.Sequential(*fc_layers)

  def forward(self, x):
    x = self.backbone(x)
    x = self.fc(x)
    # (batch,7,7,boxes*5+classes)
    return x.view(-1, self.grid_size, self.grid_size, self.num_boxes*5+self.num_classes) 
