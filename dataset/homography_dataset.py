import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

class HomographyDataset(Dataset):
    def __init__(self, frames_dir, matrices_dir, split_videos, transform=None):
        """
        Args:
            frames_dir (str): Path to the folder containing frame images.
            matrices_dir (str): Path to the folder containing homography matrices.
            split_videos (list): List of video folders to include in this dataset (train/val/test split).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.frames_dir = frames_dir
        self.matrices_dir = matrices_dir
        self.split_videos = split_videos
        self.transform = transform
        self.data = []

        # Collect all data from the specified videos
        for video in split_videos:
            video_frame_dir = os.path.join(frames_dir, video)
            video_matrix_dir = os.path.join(matrices_dir, video)
            frame_files = sorted(os.listdir(video_frame_dir))  # Sort to ensure matching order
            for frame_file in frame_files:
                frame_path = os.path.join(video_frame_dir, frame_file)
                matrix_path = os.path.join(video_matrix_dir, frame_file.replace('.jpg', '.npy'))
                if os.path.exists(matrix_path):
                    self.data.append((frame_path, matrix_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_path, matrix_path = self.data[idx]

        # Load image
        image = Image.open(frame_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load homography matrix
        homography_matrix = np.load(matrix_path).astype(np.float32)

        return image, torch.tensor(homography_matrix)
