import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# Create an instance of the convolutional neural network
input_channels = 3
output_channels = 10

net = ConvNet(input_channels, output_channels)

# Create some dummy input data
batch_size = 2
input_data = torch.randn(batch_size, input_channels, 32, 32)

# Forward pass through the network
output = net(input_data)

print("Input shape:", input_data.shape)
print("Output shape:", output.shape)
print("Network parameters:", sum(p.numel() for p in net.parameters()))
