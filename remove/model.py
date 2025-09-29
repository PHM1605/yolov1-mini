import torch 
import torch.nn as nn

architecture_config = [
  # Tuple: (number_of_filters, kernel_size, stride, padding)
  (64, 7, 2, 3), 
  "M", # max pool
  (192, 3, 1, 1),
  "M",
  (128, 1, 1, 0),
  (256, 3, 1, 1),
  (256, 1, 1, 0),
  (512, 3, 1, 1),
  "M",
  [(256, 1, 1, 0), (512, 3, 1, 1), 4], # last number: number of replicates
  (512, 1, 1, 0),
  (1024, 3, 1, 1),
  "M",
  [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
  (1024, 3, 1, 1),
  (1024, 3, 2, 1),
  (1024, 3, 1, 1),
  (1024, 3, 1, 1) 
  # Doesn't include fc layers
]

class CNNBlock(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.batchnorm = nn.BatchNorm2d(out_channels)
    self.leakyrelu = nn.LeakyReLU(0.1)
  
  def forward(self, x):
    return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
  def __init__(self, in_channels=3, **kwargs):
    super().__init__()
    self.architecture = architecture_config
    self.in_channels = in_channels 
    self.darknet = self._create_conv_layers(self.architecture)
    self.fcs = self._create_fcs(**kwargs)
  
  def forward(self, x):
    x = self.darknet(x)
    return self.fcs(torch.flatten(x, start_dim=1))
  
  def _create_conv_layers(self, architecture):
    layers = []
    in_channels = self.in_channels
    for x in architecture:
      if type(x) == tuple:
        layers += [ CNNBlock(in_channels, x[0], kernel_size=x[1], stride=x[2], padding=x[3]) ]
        in_channels = x[0]
      elif type(x) == str:
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      elif type(x) == list:
        conv1 = x[0] # tuple
        conv2 = x[1] # tuple
        repeats = x[2] # int
        for _ in range(repeats):
          layers += [ CNNBlock(in_channels, conv1[0], kernel_size=conv1[1], stride=conv1[2], padding=conv1[3])]
          layers += [ CNNBlock(conv1[0], conv2[0], kernel_size=conv2[1], stride=conv2[2], padding=conv2[3])]
          in_channels = conv2[0]
    return nn.Sequential(*layers)
  
  def _create_fcs(self, grid_size, num_boxes, num_classes):
    return nn.Sequential(
      nn.Flatten(),
      nn.Linear(1024*grid_size*grid_size, 496), # original paper uses 4096, not 496
      nn.Dropout(0.0),
      nn.LeakyReLU(0.1),
      nn.Linear(496, grid_size*grid_size*(num_classes+num_boxes*5))
    )

