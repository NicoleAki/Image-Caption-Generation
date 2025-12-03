import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    Custom CNN Encoder WITHOUT pretraining
    """
    def __init__(self, embed_size):
        super(CNNEncoder, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Image size = 224 → 112 → 56 → 28 after 3 pooling layers
        self.fc1 = nn.Linear(128 * 28 * 28, embed_size)

        # Regularization layer
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, x):
        # Layer 1
        x = self.pool(F.relu(self.conv1(x)))

        # Layer 2
        x = self.pool(F.relu(self.conv2(x)))

        # Layer 3
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected + BN
        x = self.fc1(x)
        x = self.bn(x)

        return x
