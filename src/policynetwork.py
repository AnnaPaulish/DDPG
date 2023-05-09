import torch.nn as nn

class PolicyNetwork(nn.Module):
  # actor
  def __init__(self):
    super().__init__()
    self.Lin1 = nn.Linear(3,32)
    self.Relu1 = nn.ReLU()
    self.Lin2 = nn.Linear(32,32)
    self.Relu2 = nn.ReLU()
    self.Lin3 = nn.Linear(32,1) # output a scalar value (the expected cumulative reward)
    self.Tanh = nn.Tanh()

  def forward(self,x):
    x = self.Lin1(x)
    x = self.Relu1(x)
    x = self.Lin2(x)
    x = self.Relu2(x)
    x = self.Lin3(x)
    x = self.Tanh(x)
    return x