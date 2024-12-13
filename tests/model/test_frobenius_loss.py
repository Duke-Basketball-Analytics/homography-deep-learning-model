import unittest
import torch
from model.frobenius_loss import FrobeniusConstraintLoss
from model.hyperparameters import HyperParams

class TestFrobeniusConstraintLoss(unittest.TestCase):
    def setUp(self):
        self.hparams = HyperParams()

    def test_loss_value_identity_matrix(self):
        loss_fn = FrobeniusConstraintLoss(beta=self.hparams.BETA)
        H_pred = torch.eye(3).unsqueeze(0)  # Batch of one 3x3 identity matrix
        loss = loss_fn(H_pred)
        expected_loss = ((torch.sqrt(torch.tensor(3.0)) - 1) ** 2).item()
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)

    def test_loss_value_scaled_matrix(self):
        loss_fn = FrobeniusConstraintLoss(beta=self.hparams.BETA)
        H_pred = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]])  # Frobenius norm is sqrt(12)
        expected_loss = ((torch.sqrt(torch.tensor(12.0)) - 1) ** 2).item()
        loss = loss_fn(H_pred)
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)

    def test_batch_of_matrices(self):
        loss_fn = FrobeniusConstraintLoss(beta=self.hparams.BETA)
        H_pred = torch.stack([
            torch.eye(3),  # Frobenius norm is 1
            torch.ones(3, 3)  # Frobenius norm is sqrt(9)
        ])
        expected_loss = torch.mean((torch.tensor([torch.sqrt(torch.tensor(3.0)), 3.0]) - 1) ** 2).item()
        loss = loss_fn(H_pred)
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)

    def test_invalid_input_shape(self):
        loss_fn = FrobeniusConstraintLoss(beta=self.hparams.BETA)
        H_pred = torch.tensor([1.0, 2.0, 3.0])  # Invalid shape
        with self.assertRaises(ValueError):
            loss_fn(H_pred)

    def test_beta_parameter(self):
        beta_value = 0.5
        loss_fn = FrobeniusConstraintLoss(beta=beta_value)
        self.assertEqual(loss_fn.beta, beta_value)

if __name__ == "__main__":
    unittest.main()
