import torch
import torch.nn as nn
import torch.nn.functional as F

class PCILTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(PCILTConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize PCILTs
        self.pcilt = self._create_pcilt()

    def _create_pcilt(self):
        # Assuming 8-bit quantization for activations (0-255)
        activation_range = 256
        pcilt = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, activation_range)
        for i in range(activation_range):
            pcilt[:, :, :, :, i] = self.weight * i
        return nn.Parameter(pcilt, requires_grad=False)

    def forward(self, x):
        #print(f"PCILTConv2d input shape: {x.shape}")
        
        # Quantize input to 8-bit
        x_quant = (x * 255).round().clamp(0, 255).long()
        #print(f"Quantized input shape: {x_quant.shape}")
        
        # Apply padding
        x_padded = F.pad(x_quant, (self.padding, self.padding, self.padding, self.padding))
        #print(f"Padded input shape: {x_padded.shape}")
        
        batch_size, in_channels, height, width = x_padded.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        out = torch.zeros(batch_size, self.out_channels, out_height, out_width, device=x.device)
        
        for b in range(batch_size):
            for i in range(0, height - self.kernel_size + 1, self.stride):
                for j in range(0, width - self.kernel_size + 1, self.stride):
                    patch = x_padded[b, :, i:i+self.kernel_size, j:j+self.kernel_size]
                    #print(f"Patch shape: {patch.shape}")
                    #print(f"PCILT shape: {self.pcilt.shape}")
                    
                    # Convert patch to a list of indices
                    patch_indices = patch.reshape(-1)
                    #print(f"Patch indices shape: {patch_indices.shape}")
                    #print(f"Patch indices values: {patch_indices}")
                    
                    # Use advanced indexing to get the correct PCILT values
                    pcilt_values = self.pcilt[:, :, :, :, patch_indices]
                    #print(f"PCILT values shape: {pcilt_values.shape}")
                    
                    # Sum over the appropriate dimensions
                    conv_result = pcilt_values.sum(dim=(1, 2, 3, 4))
                    #print(f"Conv result shape: {conv_result.shape}")
                    
                    out[b, :, i//self.stride, j//self.stride] = conv_result
        
        out = out + self.bias.view(1, -1, 1, 1)
        #print(f"PCILTConv2d output shape: {out.shape}")
        return out

class PCILTCNN(nn.Module):
    def __init__(self):
        super(PCILTCNN, self).__init__()
        self.conv1 = PCILTConv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = PCILTConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        
        x = self.conv1(x)
        #print(f"After conv1 shape: {x.shape}")
        
        x = F.relu(x)
        #print(f"After ReLU shape: {x.shape}")
        
        x = F.max_pool2d(x, 2)
        #print(f"After max_pool2d shape: {x.shape}")
        
        x = self.conv2(x)
        #print(f"After conv2 shape: {x.shape}")
        
        x = F.relu(x)
        #print(f"After ReLU shape: {x.shape}")
        
        x = F.max_pool2d(x, 2)
        #print(f"After max_pool2d shape: {x.shape}")
        
        x = x.view(x.size(0), -1)
        #print(f"After flatten shape: {x.shape}")
        
        x = F.relu(self.fc1(x))
        #print(f"After fc1 shape: {x.shape}")
        
        x = self.fc2(x)
        #print(f"After fc2 shape: {x.shape}")
        
        x = F.log_softmax(x, dim=1)
        #print(f"Final output shape: {x.shape}")
        
        return x

# Example usage
model = PCILTCNN()
input_tensor = torch.randn(1, 1, 28, 28)  # Example input for MNIST
print("Starting forward pass...")
output = model(input_tensor)
print("Forward pass completed.")
print(f"Final output shape: {output.shape}")