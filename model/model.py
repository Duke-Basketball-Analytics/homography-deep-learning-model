import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, hparams):
        super(CNNModel, self).__init__()
        
        # Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # Output: (32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                # Output: (32, H/4, W/4)
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # Output: (64, H/4, W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                # Output: (64, H/8, W/8)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Output: (128, H/8, W/8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # Output: (128, H/8, W/8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                  # Output: (128, H/16, W/16)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),                                         # Flatten (128 * H/16 * W/16)
            nn.Linear(128 * (hparams.IMAGE_SIZE // 16) * (hparams.IMAGE_SIZE // 16), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 8)                                     # Output: 8 parameters for 2x3 homography matrix
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x.view(-1, 8)  # Reshape to (batch_size, 8)
