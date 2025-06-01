import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Linear layers
        self.ln1 = nn.Linear(64 * 8 * 8, 128)
        self.ln2 = nn.Linear(128, 10)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        
        # 32x32 -> 16x16
        x = self.pool(x)  
        x = self.conv2(x)
        x = F.relu(x)
        
        # 16x16 -> 8x8
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 64 * 8 * 8)
        
        # Fully connected layers
        x = F.relu(self.ln1(x))
        x = self.ln2(x)
        return x
