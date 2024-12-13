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
import torch
from torch.utils.data import DataLoader

class HomographyTrainer:
    def __init__(self, hparams, dataset_cls, model_cls, criterion_cls, base_path):
        """
        Initialize the trainer with hyperparameters, dataset class, model, and other components.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hparams = hparams
        self.base_path = base_path
        
        # Initialize paths
        self.frames_dir = f"{self.base_path}/DL_frames_aug"
        self.matrices_dir = f"{self.base_path}/DL_homography_matrices"
        self.mask_dir = f"{self.base_path}/DL_masks"
        
        # Split data
        splits = self._split_data()  # Encapsulate data splitting logic
        
        # Initialize datasets and dataloaders
        self.train_loader = self._create_dataloader(dataset_cls, splits['train'], shuffle=True)
        self.val_loader = self._create_dataloader(dataset_cls, splits['val'], shuffle=False)
        self.test_loader = self._create_dataloader(dataset_cls, splits['test'], shuffle=False)
        
        # Initialize model
        self.model = model_cls(hparams).to(self.device)
        
        # Initialize loss and optimizer
        self.criterion = criterion_cls(hparams)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=hparams.LR, 
                                          weight_decay=hparams.WD)
    
    def _split_data(self):
        """
        Helper function to split the data into train, val, test.
        """
        return split_data(self.frames_dir, self.matrices_dir, 
                          train_ratio=0.7, val_ratio=0.2)

    def _create_dataloader(self, dataset_cls, split, shuffle):
        """
        Helper function to create a DataLoader for a given dataset.
        """
        dataset = dataset_cls(self.frames_dir, self.matrices_dir, self.mask_dir, split)
        return DataLoader(dataset, batch_size=self.hparams.BATCH_SIZE, shuffle=shuffle)

    def train_one_epoch(self):
        """
        Perform one epoch of training.
        """
        self.model.train()
        total_loss = 0
        for images, H_gt, points in self.train_loader:
            images, H_gt, points = images.to(self.device), H_gt.to(self.device), points.to(self.device)
            
            self.optimizer.zero_grad()
            H_pred = self.model(images)
            loss = self.criterion(H_pred, H_gt, points)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        """
        Evaluate the model on the validation/test loader.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, H_gt, points in loader:
                images, H_gt, points = images.to(self.device), H_gt.to(self.device), points.to(self.device)
                H_pred = self.model(images)
                loss = self.criterion(H_pred, H_gt, points)
                total_loss += loss.item()
        return total_loss / len(loader)

    def train(self):
        """
        Full training loop with logging and checkpointing.
        """
        best_val_loss = float('inf')
        for epoch in range(self.hparams.N_EPOCHS):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate(self.val_loader)

            print(f"Epoch {epoch+1}/{self.hparams.N_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
    
    def test(self):
        """
        Test the model on the test set.
        """
        test_loss = self.evaluate(self.test_loader)
        print(f"Test Loss: {test_loss:.4f}")
