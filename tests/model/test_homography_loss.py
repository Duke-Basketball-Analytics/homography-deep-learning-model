import unittest
import torch
from model.loss import HomographyLoss
from model.hyperparameters import HyperParams
from model.frobenius_loss import FrobeniusConstraintLoss
from model.MSE_loss import MSELoss
from model.reprojection_loss import ReprojectionLoss
from model.reprojection_weighting import AdaptiveLossWeighting

class TestHomographyLoss(unittest.TestCase):
    
    def setUp(self):
        self.hparams = HyperParams()
        self.loss_fn = HomographyLoss(self.hparams)
        self.mse_loss = MSELoss()
        self.reprojection_loss = ReprojectionLoss()
        self.frobenius_loss = FrobeniusConstraintLoss(beta = self.hparams.BETA)
        self.reprojection_scheduler = AdaptiveLossWeighting()

    
    def test_loss_components(self):
        """Test that all components contribute to the total loss correctly."""
        H_pred = torch.eye(3).unsqueeze(0) * 1.1
        H_gt = torch.eye(3).unsqueeze(0)
        points = torch.tensor([[[100, 100, 1], [200, 200, 1]]], dtype=torch.float32)
        
        total_loss = self.loss_fn(H_pred, H_gt, points)
        mse = self.mse_loss(H_pred, H_gt)
        reproj = self.reprojection_loss(H_pred, H_gt, points)
        frobenius = self.frobenius_loss(H_pred)
        
        expected_loss = mse + self.loss_fn.lambda_reproj * reproj + frobenius
        self.assertAlmostEqual(total_loss.item(), expected_loss.item(), places=6)
    
    def test_gradient_flow(self):
        """Test gradient computation for the composite loss."""
        H_pred = torch.eye(3).unsqueeze(0) * 1.1
        H_pred.requires_grad = True
        H_gt = torch.eye(3).unsqueeze(0)
        points = torch.tensor([[[100, 100, 1], [200, 200, 1]]], dtype=torch.float32)
        
        total_loss = self.loss_fn(H_pred, H_gt, points)
        total_loss.backward()
        
        self.assertIsNotNone(H_pred.grad)
        self.assertTrue(torch.all(H_pred.grad != 0))
    
    # def test_lambda_update(self):
    #     """Test dynamic lambda adjustment."""
    #     """UNDER DEVELOPMENT"""
    #     curr_loss = 0.5
    #     self.loss_fn.update_lambda(curr_loss)
    #     updated_lambda = self.loss_fn.lambda_reproj
        
    #     self.assertGreater(updated_lambda, 0)
    #     self.assertIsInstance(updated_lambda, float)

if __name__ == "__main__":
    unittest.main()
