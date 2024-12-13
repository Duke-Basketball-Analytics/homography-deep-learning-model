from dataset.dataset_utils import split_data
from dataset.homography_dataset import HomographyDataset
from torch.utils.data import DataLoader
from homography_deep_learning_model.model.frobenius_loss import frobenius_constraint_loss
from model.hyperparameters import HyperParams
from model.loss import HomographyLoss
from model.model import CNNModel
import torch
import torch.optim as optim

import os

def train(dataloader, model, criterion, optimizer, device):
    """Train full epoch on entire training dataset"""
    model.train()


def evaluate():
    """Evaluate on entire validation dataset - per epoch"""

def test():
    """Run entire test dataset"""

def training_pipeline(hparams=HyperParams()):
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
    base_path = os.path.dirname(os.path.abspath(__file__))
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

    # Initialize Loss Object
    criterion = HomographyLoss(hparams)

    # Initialize Model
    model = CNNModel(hparams)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, eps=1e-6)

    # Initialize Loss Trackers
    loss_tracking = []

    for epoch in range(hparams.N_EPOCHS):
        if epoch > 0:
            train_dataset.reset_points(seed = epoch) # data_loader queries the dataset dynamically - no need to reinitialize data loader

        for images, H_gt, points in train_loader:
            # Forward pass
            H_pred = model(images)

            # Calculate Loss 
            loss = criterion(H_pred, H_gt, points)
            loss_tracking.append(loss)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()