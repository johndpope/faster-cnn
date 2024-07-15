import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import os
from PCILTConv2d import PCILTConv2d



def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Faster Convolution Inference Through Using Pre-Calculated Lookup Tables
class PCILTCNN(nn.Module):
    def __init__(self, activation_bits=8, weight_bits=8):
        super(PCILTCNN, self).__init__()
        self.conv1 = PCILTConv2d(1, 32, kernel_size=3, stride=1, padding=1, activation_bits=activation_bits, weight_bits=weight_bits)
        self.conv2 = PCILTConv2d(32, 64, kernel_size=3, stride=1, padding=1, activation_bits=activation_bits, weight_bits=weight_bits)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #print("Starting conv1")
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #print(f"After conv1 and pooling shape: {x.shape}")
        
        #print("Starting conv2")
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #print(f"After conv2 and pooling shape: {x.shape}")
        
        x = x.view(x.size(0), -1)
        #print(f"After flattening shape: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
# Example usage
#print("Creating model...")
model = PCILTCNN(activation_bits=4, weight_bits=4)  # Using 4-bit quantization as an example
print_memory_usage()

#print("Creating input tensor...")
input_tensor = torch.randn(1, 1, 28, 28)  # Example input for MNIST
print_memory_usage()

#print("Starting forward pass...")
output = model(input_tensor)
print("Forward pass completed.")
print(f"Final output shape: {output.shape}")
print_memory_usage()