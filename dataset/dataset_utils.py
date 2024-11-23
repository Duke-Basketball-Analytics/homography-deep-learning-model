import os
import random

def split_data(frames_dir, matrices_dir, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    Splits videos into Train, Validation, and Test datasets.

    Args:
        frames_dir (str): Path to the folder containing frame images.
        matrices_dir (str): Path to the folder containing homography matrices.
        train_ratio (float): Proportion of videos to use for training.
        val_ratio (float): Proportion of videos to use for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' lists of video names.
    """
    random.seed(seed)
    videos = sorted(os.listdir(frames_dir))
    
    # Shuffle videos for randomness
    random.shuffle(videos)

    # Compute split indices
    num_videos = len(videos)
    train_split = int(num_videos * train_ratio)
    val_split = int(num_videos * (train_ratio + val_ratio))

    # Split data
    splits = {
        'train': videos[:train_split],
        'val': videos[train_split:val_split],
        'test': videos[val_split:]
    }

    return splits
