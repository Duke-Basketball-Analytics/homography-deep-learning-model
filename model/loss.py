import torch
import torch.nn as nn
from reprojection_loss import ReprojectionLoss
from frobenius_loss import FrobeniusConstraintLoss
from reprojection_weighting import AdaptiveLossWeighting
from MSE_loss import MSELoss

class HomographyLoss(nn.Module):
    def __init__(self, hparams):
        super(HomographyLoss, self).__init__()
        self.mse_loss = MSELoss()
        self.reprojection_loss = ReprojectionLoss()
        self.frobenius_loss = FrobeniusConstraintLoss(beta = hparams.BETA)
        self.reprojection_scheduler = AdaptiveLossWeighting()
        self.lambda_reproj = 0

    def forward(self, H_pred, H_gt, points):
        mse = self.mse_loss(H_pred, H_gt)
        reproj = self.reprojection_loss(H_pred, H_gt, points)
        frobenius = self.frobenius_loss(H_pred)
        return mse + self.lambda_reproj*reproj + frobenius

    def update_lambda(self, curr_loss):
        self.lambda_reproj = self.reprojection_scheduler.update_lambda(curr_loss) # Use current loss to calculate weight of reprojection loss component
