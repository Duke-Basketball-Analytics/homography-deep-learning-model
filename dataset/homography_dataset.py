import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from point_selection import sample_evenly_spaced_points

class HomographyDataset(torch.utils.data.Dataset):
    """This is the dataset class used for training/validation/test"""
    def __init__(self, frames_dir, matrices_dir, mask_dir, video_list, transform=None, num_points=25, grid_size=(12,12)):
        self.frames_dir = frames_dir
        self.matrices_dir = matrices_dir
        self.mask_dir = mask_dir
        self.video_list = video_list
        self.transform = transform
        self.num_points = num_points
        self.grid_size = grid_size
        self.data = []
        self.points = {}  # Store points for each image
        self.seed = None

        # Collect data paths
        for video in video_list:
            frame_vid_dir = os.path.join(frames_dir, video)
            matrix_vid_dir = os.path.join(matrices_dir, video)
            mask_vid_dir = os.path.join(mask_dir, video)
            frame_files = sorted(os.listdir(frame_vid_dir))
            for frame_file in frame_files:
                frame_path = os.path.join(frame_vid_dir, frame_file)
                matrix_path = os.path.join(matrix_vid_dir, frame_file.replace('.jpg', '.npy'))
                mask_path = os.path.join(mask_vid_dir, frame_file.replace('.jpg', '.npy'))
                if os.path.exists(matrix_path) and os.path.exists(mask_path):
                    self.data.append((frame_path, matrix_path, mask_path))

    def reset_points(self, seed):
        """Generate new random points for each image in the dataset."""
        for idx, (frame_path, matrix_path, mask_path) in enumerate(self.data):
            mask = np.load(mask_path)
            sampled_points = sample_evenly_spaced_points(mask, self.num_points, self.grid_size, seed=seed)
            self.points[idx] = torch.tensor(sampled_points, dtype=torch.float32)

    def reset_seed(self, seed):
        """
        Reset the seed after every epoch.
        Used to randomize the points used in reprojection calculation after every epoch.
        """
        self.seed = seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        """
        Function indexes data based on the batch configuration of the data_loader
        Data is stored as paths instead of actual data to reduce memory.
        """
        frame_path, matrix_path, mask_path = self.data[idx]

        # Load image
        image = Image.open(frame_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load mask
        mask = np.load(mask_path)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Add mask as an additional channel
        image_with_mask = torch.cat((image, mask.unsqueeze(0)), dim=0)

        # Load ground truth homography matrix
        homography_matrix = np.load(matrix_path).astype(np.float32)
        homography_matrix = torch.tensor(homography_matrix, dtype=torch.float32)

        # Get precomputed points
        sampled_points = sample_evenly_spaced_points(mask, self.num_points, self.grid_size, seed=self.seed)
        sampled_points = torch.tensor(sampled_points, dtype=torch.float32)

        return image_with_mask, homography_matrix, sampled_points