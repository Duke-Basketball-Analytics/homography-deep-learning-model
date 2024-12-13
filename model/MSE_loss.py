import torch
import torch.nn as nn

class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, H_pred, H_gt):

        if H_pred.ndim != 3 or H_pred.shape[1:] != (3, 3):
            raise ValueError("H_pred must have shape [batch_size, 3, 3].")
        if H_gt.ndim != 3 or H_gt.shape[1:] != (3, 3):
            raise ValueError("H_gt must have shape [batch_size, 3, 3].")

        return torch.nn.functional.mse_loss(H_pred, H_gt)
