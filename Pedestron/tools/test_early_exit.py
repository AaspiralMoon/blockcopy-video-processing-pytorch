import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyExitCNN(nn.Module):
    def __init__(self):
        super(EarlyExitCNN, self).__init__()
        self.layer1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        outputs = torch.tensor([])  # Initialize an empty tensor for collecting outputs
        
        # First layer processing
        x1 = F.relu(self.layer1(x))
        # Assume first 3 inputs exit after layer 1
        outputs = torch.cat((outputs, x1[:3]), 0)
        x1 = x1[3:]  # Continue processing the rest
        
        # Second layer processing
        x2 = F.relu(self.layer2(x1))
        # Assume next 2 inputs exit after layer 2
        outputs = torch.cat((outputs, x2[:2]), 0)
        x2 = x2[2:]  # Continue processing the rest
        
        # Third layer processing for the remaining inputs
        x3 = F.relu(self.layer3(x2))
        outputs = torch.cat((outputs, x3), 0)
        
        return outputs

# Example usage
model = EarlyExitCNN()
# Assuming dummy input data: batch size of 10, 3 channels, 64x64 pixels
dummy_input = torch.randn(10, 3, 64, 64)
output = model(dummy_input)
print(output.size())
