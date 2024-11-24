import torch
import torch.nn as nn

class FrobeniusConstraintLoss(nn.Module):

    def __init__(self, beta):
        super(FrobeniusConstraintLoss, self).__init__()
        self.beta = beta #

    def forward(self, H_pred):
        norm = torch.sqrt(torch.sum(H_pred**2, dim=(1, 2)))
        return torch.mean((norm - 1)**2)
