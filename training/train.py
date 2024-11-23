from dataset.dataset_utils import split_data
from dataset.homography_dataset import HomographyDataset
from torch.utils.data import DataLoader

# Paths to data
frames_dir = "DL_homography/DL_frames_aug"
matrices_dir = "DL_homography/DL_homography_matrices"

# Split data
splits = split_data(frames_dir, matrices_dir, train_ratio=0.7, val_ratio=0.2)

# Create Datasets
train_dataset = HomographyDataset(frames_dir, matrices_dir, splits['train'])
val_dataset = HomographyDataset(frames_dir, matrices_dir, splits['val'])
test_dataset = HomographyDataset(frames_dir, matrices_dir, splits['test'])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
