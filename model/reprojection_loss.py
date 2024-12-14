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
        if H_pred.ndim != 3 or H_pred.shape[1:] != (3, 3):
            raise ValueError("H_pred must have shape [batch_size, 3, 3].")
        if H_gt.ndim != 3 or H_gt.shape[1:] != (3, 3):
            raise ValueError("H_gt must have shape [batch_size, 3, 3].")
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError("points must have shape [batch_size, num_points, 3].")

        # Transform points using predicted and ground truth homographies
        pred_points = torch.bmm(points, H_pred.transpose(1, 2))  # [batch_size, N, 3]
        gt_points = torch.bmm(points, H_gt.transpose(1, 2))      # [batch_size, N, 3]

        # Normalize to convert to non-homogeneous coordinates
        pred_points = pred_points / pred_points[:, :, -1:].clamp(min=1e-8)
        gt_points = gt_points / gt_points[:, :, -1:].clamp(min=1e-8)

        # Compute Euclidean distance between predicted and ground truth points
        error = torch.sqrt(torch.sum((pred_points[:, :, :2] - gt_points[:, :, :2]) ** 2, dim=-1))  # [batch_size, N]

        # Average reprojection error across all points in the batch
        return self.reprojection_weighting(torch.mean(error))
    
    def reprojection_weighting(self, reproj_loss):
        """
        Inverse exponential decay function.
        Maps the reprojection loss into a scaling factor between 0 and 1.

        f(x) = (B - A) * (1 - exp(-k * x)) + A

        where:
        x: average reprojection error (reproj_loss)
        A: Intercept (x=0, y=0.003)
        B: Upper Bound Asymptote (y=1)
        k: Rate of Increase parameter

        Parameter k was calculated so that f(2240) = 0.997

        x = 2240 represents an average reprojection error of 10x the image coordinate system (224x224)
        At this point, we use the 3 standard deviation percentage to ensure the effects of reprojection error are felt.
        """
        A = 0.003
        B = 1.0
        k = 0.0025920261
        return torch.tensor((B - A) * (1 - torch.exp(-k * reproj_loss)) + A)
