from dataset.dataset_utils import split_data
from dataset.homography_dataset import HomographyDataset
from torch.utils.data import DataLoader
import os

def train():
    """Train full epoch on entire training dataset"""

def evaluate():
    """Evaluate on entire validation dataset - per epoch"""

def test():
    """Run entire test dataset"""

def training_pipeline():
    """
    1. Train/Val/Test Split
    2. Create datasets
    3. Create dataloaders
    4. Initialize model
    5. Initialize Optimizer and Criterion
    6. Iterate through epochs calling train() and evaluate()
    7. Run test()
    """

    # Paths to data
    base_path = "/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography"
    frames_dir = os.path.join(base_path, "DL_frames_aug")
    matrices_dir = os.path.join(base_path, "DL_homography_matrices")
    mask_dir = os.path.join(base_path, "DL_masks")

    # Split data
    splits = split_data(frames_dir, matrices_dir, train_ratio=0.7, val_ratio=0.2)

    # Create Datasets
    train_dataset = HomographyDataset(frames_dir, matrices_dir, mask_dir, splits['train'])
    val_dataset = HomographyDataset(frames_dir, matrices_dir, mask_dir, splits['val'])
    test_dataset = HomographyDataset(frames_dir, matrices_dir, mask_dir, splits['test'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



    # Training loop
    for epoch in range(num_epochs):
        # Regenerate points for the new epoch
        dataset.reset_points()

        # Iterate through DataLoader
        for images, H_gt, points in dataloader:
            # Forward pass
            H_pred = model(images)
