import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

class PCILTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation_bits=8, weight_bits=8):
        super(PCILTConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize PCILTs
        self.pcilt = self._create_pcilt()
        #print(f"PCILT shape: {self.pcilt.shape}")
        print_memory_usage()

    def _create_pcilt(self):
        activation_range = 2 ** self.activation_bits
        weight_range = 2 ** self.weight_bits
        
        # Use a smaller data type to reduce memory usage
        pcilt = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, activation_range, dtype=torch.int16)
        
        # Quantize weights
        quantized_weights = self._quantize_weights(self.weight, weight_range)
        
        for i in range(activation_range):
            pcilt[:, :, :, :, i] = (quantized_weights * i).to(torch.int16)
        return nn.Parameter(pcilt, requires_grad=False)

    def _quantize_weights(self, weights, weight_range):
        return (weights * (weight_range - 1)).round().clamp(0, weight_range - 1)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        # Quantize input to specified bit depth
        x_quant = (x * (2**self.activation_bits - 1)).round().clamp(0, 2**self.activation_bits - 1).long()
        #print(f"Quantized input shape: {x_quant.shape}")
        
        # Apply padding
        x_padded = F.pad(x_quant, (self.padding, self.padding, self.padding, self.padding))
        #print(f"Padded input shape: {x_padded.shape}")
        
        batch_size, _, height, width = x_padded.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        out = torch.zeros(batch_size, self.out_channels, out_height, out_width, device=x.device)
        
        for b in range(batch_size):
            for i in range(0, height - self.kernel_size + 1, self.stride):
                for j in range(0, width - self.kernel_size + 1, self.stride):
                    patch = x_padded[b, :, i:i+self.kernel_size, j:j+self.kernel_size]
                    #print(f"Patch shape: {patch.shape}")
                    pcilt_values = self.pcilt[:, :, :, :, patch]
                    #print(f"PCILT values shape: {pcilt_values.shape}")
                    conv_result = pcilt_values.to(torch.float32).sum(dim=(1, 2, 3, 4, 5, 6))
                    #print(f"Conv result shape: {conv_result.shape}")
                    #print(f"Conv result: {conv_result}")
                    out[b, :, i//self.stride, j//self.stride] = conv_result
        
        out = out + self.bias.view(1, -1, 1, 1)
        #print(f"Output shape: {out.shape}")
        print_memory_usage()
        return out

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