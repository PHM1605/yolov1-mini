import torch.nn as nn 

class YoloV1Tiny(nn.Module):
  def __init__(self, S=7, B=2, C=2):
    super().__init__()
    self.S, self.B, self.C = S, B, C 
    self.features = nn.Sequential(
      nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2,2), # (16,224,224)
      nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2,2), # (32,112,112)
      nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2,2), # (64,56,56)
      nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2,2), # (128,28,28)
      nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((7,7)) # (256,7,7)
    )
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(256*7*7, 512), nn.ReLU(),
      nn.Linear(512, S*S*(B*5+C))
    )

  def forward(self, x):
    x = self.features(x)
    x = self.fc(x)
    return x.view(-1, self.S, self.S, self.B*5+self.C) # (batch,S,S,B*5+C)
