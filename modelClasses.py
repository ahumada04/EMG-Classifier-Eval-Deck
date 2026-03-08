import torch
import torch.nn as nn
import torch.nn.functional as F

# Erick
class CNN_FANet(nn.Module):
    def __init__(self, num_classes=6):
        super(CNN_FANet, self).__init__()
        
        # Layer 1: Look for local patterns across the 8 channels
        # Input shape: (Batch, 8, Window_Size)
        # 8 due to the 8 channels of actual input data
        # 32 being a simple power of 2, big enough to capture complexity, small enough for laptop lol
        # 3 length of "filter" itself, smaller given high noise data like EMG
        # 1 gives data length same after convolution
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
        
        # Layer 2: Deeper features
        # same as above, but with a second layer lol
        # higher scale given # of outputs from prev
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Layer 3: Channel Attention (The 'Adaptable' part of FANet)
        # Effictivly maps sensor vars to action
        # Ex: association between high activity in sensor 1 & 2 and fist gesture
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(64, 16, 1),
            nn.ReLU(),
            nn.Conv1d(16, 64, 1),
            nn.Sigmoid()
        )
        
        # Final Classifier
        self.fc = nn.Linear(64, num_classes)
    
    # function to allow forward pass through to begin training
    def forward(self, x):
        # 1. Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # 2. Apply Attention
        weights = self.attention(x)
        x = x * weights
        
        # 3. Global Average Pooling (reduces time dimension to 1)
        x = torch.mean(x, dim=2) 
        
        # 4. Classify
        x = self.fc(x)
        return x
    

# Your class goes here :D
