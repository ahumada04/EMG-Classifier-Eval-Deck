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
    

# Tony
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=6, dropout: float = 0.3, bidirectional: bool = True):
        """
        Initialize the LSTMClassifier with the specified parameters for the LSTM layers and the final classifier layer.
        """

        # Call the superclass constructor to properly initialize the nn.Module
        super().__init__()

        # Store the parameters as instance variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional


        # Define the LSTM layer with the specified input size, hidden size, number of layers, dropout, and bidirectionality 
        # The batch_first parameter is set to True to indicate that the input tensors will have the batch dimension as the first dimension
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, dropout = dropout if num_layers > 1 else 0.0, bidirectional = bidirectional)

        # Calculate the output size of the LSTM layer
        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        # Define the final classifier layer as a sequential model consisting of a dropout layer followed by a linear layer that maps the LSTM output to the number of classes
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_output_size, num_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the LSTMClassifier. 
        """

        # Pass the input tensor x through the LSTM layer to obtain the output features and the hidden state 
        # Only need the hidden state (h_n) for classification, so ignore the output and the cell state (c_n)
        _, (h_n, _) = self.lstm(x)

        # Depending on whether the LSTM is bidirectional or not, need to extract the appropriate hidden state(s) to use as features for the classifier
        # If the LSTM is bidirectional, concatenate the final hidden states from both directions
        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            features = torch.cat((h_forward, h_backward), dim = 1)

        # If the LSTM is not bidirectional, take the final hidden state from the last layer
        else:
            features = h_n[-1]
        
        # Pass the extracted features through the classifier layer to obtain the logits for each class
        logits = self.head(features)
        return logits