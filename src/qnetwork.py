import torch.nn as nn

class QNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.Lin1 = nn.Linear(4,32)
    self.Relu1 = nn.ReLU()
    self.Lin2 = nn.Linear(32,32)
    self.Relu1 = nn.ReLU() # add
    self.Lin3 = nn.Linear(32,1) # output a scalar value (the expected cumulative reward)

  def forward(self,x):
    x = self.Lin1(x)
    x = self.Relu1(x)
    x = self.Lin2(x)
    x = self.Relu1(x)
    x = self.Lin3(x)
    return x