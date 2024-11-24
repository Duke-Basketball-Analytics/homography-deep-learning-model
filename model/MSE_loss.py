import torch
import torch.nn as nn

class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target):
        return torch.nn.functional.mse_loss(pred, target)
