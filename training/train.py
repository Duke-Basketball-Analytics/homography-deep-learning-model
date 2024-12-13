from dataset.dataset_utils import split_data
from dataset.homography_dataset import HomographyDataset
from torch.utils.data import DataLoader
from model.hyperparameters import HyperParams
from model.loss import HomographyLoss
from model.model import CNNModel
import torch
import torch.optim as optim
import os
import torch
from torch.utils.data import DataLoader

class HomographyTrainer:
    def __init__(self):
        """
        Initialize the trainer with hyperparameters, dataset class, model, and other components.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hparams = HyperParams()
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize paths
        self.frames_dir = os.path.abspath(os.path.join(self.base_path, 'DL_frames_aug'))
        self.matrices_dir = os.path.abspath(os.path.join(self.base_path, 'DL_homography_matrices'))
        self.mask_dir = os.path.abspath(os.path.join(self.base_path, 'DL_masks'))
        
        # Split data
        splits = self._split_data()  # Encapsulate data splitting logic

        # Create datasets
        self.train_dataset = self._create_dataset(splits['train'])
        self.val_dataset = self._create_dataset(splits['val'])
        self.test_dataset = self._create_dataset(splits['test'])

        # Create data loaders
        self.train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_dataset, shuffle=False)
        self.test_loader = self._create_dataloader(self.test_dataset, shuffle=False)
        
        # Initialize model
        self.model = CNNModel(self.hparams).to(self.device)
        
        # Initialize loss and optimizer
        self.criterion = HomographyLoss(self.hparams)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.hparams.LR, 
                                          weight_decay=self.hparams.WD)
    
    def _split_data(self):
        """
        Helper function to split the data into train, val, test.
        """
        return split_data(self.frames_dir, self.matrices_dir, 
                          train_ratio=0.7, val_ratio=0.2)
    
    def _create_dataset(self, split):
        """
        Helper function to create a Homography Dataset.
        """
        return HomographyDataset(self.frames_dir, self.matrices_dir, self.mask_dir, split)

    def _create_dataloader(self, dataset, shuffle):
        """
        Helper function to create a DataLoader for a given dataset.
        """

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
            self.train_dataset.reset_seed(seed=epoch)
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
