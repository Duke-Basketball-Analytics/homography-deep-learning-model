import torch
import torch.nn as nn
import torch.nn.functional as F

class ReprojectionLoss(nn.Module):
    def __init__(self):
        super(ReprojectionLoss, self).__init__()
    
    def forward(self, H_pred, H_gt, points):
        """
        Args:
            H_pred (torch.Tensor): Predicted homography matrices of shape [batch_size, 3, 3].
            H_gt (torch.Tensor): Ground truth homography matrices of shape [batch_size, 3, 3].
            points (torch.Tensor): Input points to transform, shape [batch_size, N, 3]
                                    where each point is homogeneous (x, y, 1).

        Returns:
            torch.Tensor: Reprojection error for the batch.
        """
        # Transform points using predicted and ground truth homographies
        pred_points = torch.bmm(points, H_pred.transpose(1, 2))  # [batch_size, N, 3]
        gt_points = torch.bmm(points, H_gt.transpose(1, 2))      # [batch_size, N, 3]

        # Normalize to convert to non-homogeneous coordinates
        pred_points = pred_points / pred_points[:, :, -1:].clamp(min=1e-8)
        gt_points = gt_points / gt_points[:, :, -1:].clamp(min=1e-8)

        # Compute Euclidean distance between predicted and ground truth points
        error = torch.sqrt(torch.sum((pred_points[:, :, :2] - gt_points[:, :, :2]) ** 2, dim=-1))  # [batch_size, N]

        # Average reprojection error across all points in the batch
        return torch.mean(error)
