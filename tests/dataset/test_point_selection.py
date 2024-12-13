from dataset.point_selection import sample_evenly_spaced_points
import unittest
import numpy as np

class TestSampleEvenlySpacedPoints(unittest.TestCase):
    def setUp(self):
        # Create a sample binary mask for testing
        self.mask = np.zeros((240, 240), dtype=int)
        self.mask[4:20, 4:20] = 1  # Define a court region in the middle

    def test_output_shape(self):
        num_points = 10
        grid_size = (6, 6)
        seed = 42

        points = sample_evenly_spaced_points(self.mask, num_points, seed, grid_size)
        
        # Check if the output shape is correct
        self.assertEqual(points.shape, (num_points, 3))

    def test_points_within_court(self):
        num_points = 20
        grid_size = (4, 4)
        seed = 123

        points = sample_evenly_spaced_points(self.mask, num_points, seed, grid_size)

        # Check if all sampled points are within the court region
        for point in points:
            x, y, _ = point
            self.assertEqual(self.mask[int(y), int(x)], 1)

    def test_random_seed_reproducibility(self):
        num_points = 15
        grid_size = (6, 6)
        seed = 42

        points1 = sample_evenly_spaced_points(self.mask, num_points, seed, grid_size)
        points2 = sample_evenly_spaced_points(self.mask, num_points, seed, grid_size)

        # Check if results are reproducible with the same seed
        np.testing.assert_array_equal(points1, points2)

    def test_fewer_points_than_requested(self):
        num_points = 50
        grid_size = (12, 12)
        seed = 7

        # Use a smaller mask to simulate fewer valid points
        small_mask = np.zeros((24, 24), dtype=int)
        small_mask[10:14, 10:14] = 1

        points = sample_evenly_spaced_points(small_mask, num_points, seed, grid_size)

        # Check if output has the correct number of points
        self.assertEqual(points.shape[0], num_points)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            sample_evenly_spaced_points(np.array([]), 10, 42, (4, 4))

        with self.assertRaises(ValueError):
            sample_evenly_spaced_points(self.mask, -5, 42, (4, 4))

        with self.assertRaises(ValueError):
            sample_evenly_spaced_points(self.mask, 10, 42, (0, 0))

if __name__ == "__main__":
    unittest.main()
