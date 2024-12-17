import numpy as np
import torch

def sample_evenly_spaced_points(mask, num_points, seed, grid_size=(12,12)):
    """
    Samples evenly spaced random points from the court region of the mask.
    
    Args:
        mask (np.ndarray): Binary mask of shape [H, W] (1 = court, 0 = non-court).
        num_points (int): Total number of points to sample.
        grid_size (tuple): Number of grid cells in (rows, cols).
    
    Returns:
        np.ndarray: Array of sampled points in homogeneous coordinates [x, y, 1].
    """
    # Set seed for reproducability
    np.random.seed(seed)

    # Mask dimensions
    # Mask shape has to be 2 dimensions
    h, w = mask.shape

    # Input validation
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a 2D numpy array.")
    if not isinstance(num_points, int) or num_points <= 0:
        raise ValueError("num_points must be a positive integer.")
    if not (isinstance(grid_size, tuple) and len(grid_size) == 2 and all(isinstance(x, int) and x > 0 for x in grid_size)):
        raise ValueError("grid_size must be a tuple of two positive integers.")
    
    # Divide mask into grid cells
    rows, cols = grid_size
    row_step, col_step = h // rows, w // cols
    
    # List to store sampled points
    sampled_points = []
    
    # Randomly sample points from each grid cell
    for i in range(rows):
        for j in range(cols):
            # Get the cell boundaries
            row_start, row_end = i * row_step, min((i + 1) * row_step, h)
            col_start, col_end = j * col_step, min((j + 1) * col_step, w)
            
            # Get court points in the current cell
            cell_indices = np.argwhere(mask[row_start:row_end, col_start:col_end] == 1)
            
            # If there are valid points in the cell
            if len(cell_indices) > 0:
                # Add offset to get global coordinates
                cell_indices += [row_start, col_start]
                
                # Randomly select one point from the cell
                sampled_point = cell_indices[np.random.choice(len(cell_indices))]
                sampled_points.append(sampled_point)
    
    # If we have fewer points than required, adjust
    sampled_points = np.array(sampled_points)
    if len(sampled_points) > num_points:
        # Randomly select a subset of the points
        sampled_points = sampled_points[np.random.choice(len(sampled_points), num_points, replace=False)]
    elif len(sampled_points) < num_points:
        # Randomly duplicate points if fewer than required
        additional_points = sampled_points[np.random.choice(len(sampled_points), num_points - len(sampled_points), replace=True)]
        sampled_points = np.vstack([sampled_points, additional_points])
    
    # Convert to homogeneous coordinates [x, y, 1]
    sampled_points = np.hstack([sampled_points, np.ones((len(sampled_points), 1))])
    
    return sampled_points
