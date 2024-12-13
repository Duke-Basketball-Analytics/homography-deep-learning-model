import unittest
from model.reprojection_weighting import AdaptiveLossWeighting

class TestAdaptiveLossWeighting(unittest.TestCase):

    def setUp(self):
        self.loss_weighting = AdaptiveLossWeighting(epsilon_v=1e-3, epsilon_a=1e-4, max_lambda=1, step_size=0.05)

    def test_initialization(self):
        """Test that the class initializes correctly."""
        self.assertIsNone(self.loss_weighting.prev_loss, "prev_loss should be None on initialization.")
        self.assertEqual(self.loss_weighting.prev_velocity, 0, "prev_velocity should be initialized to 0.")
        self.assertEqual(self.loss_weighting.lambda_val, 0, "lambda_val should be initialized to 0.")

    def test_lambda_increment(self):
        """Test that lambda_val increments correctly when thresholds are satisfied."""
        self.loss_weighting.prev_loss = 1.0
        self.loss_weighting.prev_velocity = 0.001  # Just below epsilon_a threshold

        new_lambda = self.loss_weighting.update_lambda(1.0005)  # Velocity and acceleration satisfy thresholds
        self.assertAlmostEqual(new_lambda, 0.05, "lambda_val should increment by step_size.", places=6)

    def test_no_increment(self):
        """Test that lambda_val does not increment when thresholds are not satisfied."""
        self.loss_weighting.prev_loss = 1.0
        self.loss_weighting.prev_velocity = 0.01  # Above epsilon_a threshold

        new_lambda = self.loss_weighting.update_lambda(1.02)  # Velocity and acceleration exceed thresholds
        self.assertEqual(new_lambda, 0, "lambda_val should not increment when thresholds are not met.")

    # def test_edge_case_constant_loss(self):
    #     """Test that lambda_val does not increment when the loss is constant."""
    #     self.loss_weighting.prev_loss = 1.0
    #     new_lambda = self.loss_weighting.update_lambda(1.0)  # No change in loss
    #     self.assertEqual(new_lambda, 0, "lambda_val should not increment when the loss is constant.")

    def test_oscillatory_behavior(self):
        """Test the behavior of lambda_val for oscillating losses."""
        losses = [1.0, 1.01, 1.0, 1.01, 1.0]  # Oscillating pattern
        self.loss_weighting.prev_loss = losses[0]

        for loss in losses[1:]:
            self.loss_weighting.update_lambda(loss)

        # Ensure lambda_val does not increment uncontrollably due to oscillations
        self.assertLessEqual(self.loss_weighting.lambda_val, self.loss_weighting.max_lambda)

if __name__ == "__main__":
    unittest.main()


    """UNDER DEVELOPMENT"""
