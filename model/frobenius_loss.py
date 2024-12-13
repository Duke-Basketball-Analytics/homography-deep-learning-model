import torch
import torch.nn as nn

# class FrobeniusConstraintLoss(nn.Module):

#     def __init__(self, beta):
#         super(FrobeniusConstraintLoss, self).__init__()
#         self.beta = beta #

#     def forward(self, H_pred):
#         norm = torch.sqrt(torch.sum(H_pred**2, dim=(1, 2)))
#         return torch.mean((norm - 1)**2)
    
class FrobeniusConstraintLoss(nn.Module):

    def __init__(self, beta):
        super(FrobeniusConstraintLoss, self).__init__()
        self.beta = beta

    def forward(self, H_pred):
        # Validate input shape
        if H_pred.ndim != 3:
            raise ValueError("H_pred must be a 3D tensor with shape (batch_size, rows, cols).")
        if H_pred.shape[1] != 3 or H_pred.shape[2] != 3:
            raise ValueError(f"Homography matrix is not 3x3: Dimensions {H_pred.shape[1]}x{H_pred.shape[2]}")
        
        # Calculate Frobenius norm for each matrix in the batch
        norm = torch.sqrt(torch.sum(H_pred**2, dim=(1, 2)))

        # Return the mean squared difference from 1
        return torch.mean((norm - 1)**2)