import torch.nn as nn

class CriticModel(nn.Module):
  def __init__(self, model, *args, **kwargs):
    super(CriticModel, self).__init__()
    self.model = model(*args, **kwargs)

  def forward(self, state):
    return self.model(state)
