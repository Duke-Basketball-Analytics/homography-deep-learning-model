import unittest
import torch
from your_module import ReprojectionLoss  # Replace 'your_module' with the actual module name

class TestReprojectionLoss(unittest.TestCase):

    def setUp(self):
        self.loss_fn = ReprojectionLoss()

    def test_zero_loss_for_identity(self):
        """Test that the loss is zero when H_pred equals H_gt."""
        batch_size = 2
        num_points = 4

        H_gt = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)  # Ground truth as identity matrices
        H_pred = H_gt.clone()  # Predicted is same as ground truth

        points = torch.tensor([
            [[0, 0, 1], [224, 0, 1], [0, 224, 1], [224, 224, 1]],
            [[50, 50, 1], [100, 50, 1], [50, 100, 1], [100, 100, 1]]
        ], dtype=torch.float32)  # Homogeneous points of shape [batch_size, num_points, 3]

        loss = self.loss_fn(H_pred, H_gt, points)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_nonzero_loss(self):
        """Test that the loss is non-zero when H_pred is different from H_gt."""
        batch_size = 1
        num_points = 4

        H_gt = torch.eye(3).unsqueeze(0)  # Ground truth as identity matrix
        H_pred = torch.tensor([
            [[1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.0]]
        ])  # Scaled matrix

        points = torch.tensor([
            [[0, 0, 1], [224, 0, 1], [0, 224, 1], [224, 224, 1]]
        ], dtype=torch.float32)  # Homogeneous points of shape [batch_size, num_points, 3]

        loss = self.loss_fn(H_pred, H_gt, points)
        self.assertGreater(loss.item(), 0.0)

    def test_invalid_shapes(self):
        """Test that the function raises errors for invalid input shapes."""
        batch_size = 1
        num_points = 4

        H_gt = torch.eye(3).unsqueeze(0)  # Ground truth as identity matrix
        H_pred = torch.eye(3).unsqueeze(0)  # Predicted as identity matrix

        points_invalid = torch.tensor([[0, 0, 1], [1, 0, 1]], dtype=torch.float32)  # Missing batch dimension

        with self.assertRaises(RuntimeError):
            self.loss_fn(H_pred, H_gt, points_invalid)

    def test_batch_loss_computation(self):
        """Test loss computation for a batch of matrices and points."""
        batch_size = 2
        num_points = 4

        H_gt = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)  # Ground truth as identity matrices
        H_pred = torch.tensor([
            [[1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.0]],
            [[1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.0]]
        ])  # Slightly scaled matrices

        points = torch.tensor([
            [[0, 0, 1], [224, 0, 1], [0, 224, 1], [224, 224, 1]],
            [[50, 50, 1], [100, 50, 1], [50, 100, 1], [100, 100, 1]]
        ], dtype=torch.float32)  # Homogeneous points

        loss = self.loss_fn(H_pred, H_gt, points)
        self.assertGreater(loss.item(), 0.0)

if __name__ == "__main__":
    unittest.main()
