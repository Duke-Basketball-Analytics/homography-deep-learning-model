import unittest
import torch
from model.MSE_loss import MSELoss  # Replace 'your_module' with the actual module name

class TestMSELoss(unittest.TestCase):

    def setUp(self):
        self.loss_fn = MSELoss()

    def test_zero_loss(self):
        """Test that the loss is zero when predictions match the target."""
        H_pred = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])  # Identity matrix
        H_gt = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])  # Same as prediction
        loss = self.loss_fn(H_pred, H_gt)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_positive_loss(self):
        """Test that the loss is positive when predictions differ from the target."""
        H_pred = torch.tensor([[[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])  # Slightly off identity matrix
        H_gt = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])  # Identity matrix
        loss = self.loss_fn(H_pred, H_gt)
        expected_loss = torch.mean((H_pred - H_gt) ** 2).item()
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)

    def test_batch_loss(self):
        """Test loss computation for batched inputs."""
        H_pred = torch.tensor([
            [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.9]]
        ])  # Batch of slightly off identity matrices
        H_gt = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ])  # Batch of identity matrices
        loss = self.loss_fn(H_pred, H_gt)
        expected_loss = torch.mean((H_pred - H_gt) ** 2).item()
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)

    def test_different_shapes(self):
        """Test that the function raises an error for mismatched shapes."""
        H_pred = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        H_gt = torch.tensor([1.0, 0.0, 0.0])  # Invalid shape
        with self.assertRaises(ValueError):
            self.loss_fn(H_pred, H_gt)

    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        H_pred = torch.tensor([[[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], requires_grad=True)
        H_gt = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        loss = self.loss_fn(H_pred, H_gt)
        loss.backward()
        expected_grad = 2 * (H_pred - H_gt) / H_pred.numel()  # d(MSE)/d(pred)
        self.assertTrue(torch.allclose(H_pred.grad, expected_grad, atol=1e-6))

if __name__ == "__main__":
    unittest.main()
