import torch
import torch.nn as nn
import torch.nn.functional as F
import pcilt_conv_cuda

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

    def _create_pcilt(self):
        activation_range = 2 ** self.activation_bits
        weight_range = 2 ** self.weight_bits
        
        pcilt = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, activation_range, dtype=torch.float32)
        
        # Quantize weights
        quantized_weights = self._quantize_weights(self.weight, weight_range)
        
        for i in range(activation_range):
            pcilt[:, :, :, :, i] = quantized_weights * i
        return nn.Parameter(pcilt, requires_grad=False)

    def _quantize_weights(self, weights, weight_range):
        return (weights * (weight_range - 1)).round().clamp(0, weight_range - 1)

    def forward(self, x):
        # Quantize input to specified bit depth
        x_quant = (x * (2**self.activation_bits - 1)).round().clamp(0, 2**self.activation_bits - 1).long()
        
        # Use CUDA implementation
        out = pcilt_conv_cuda.pcilt_conv_cuda(
            x_quant.float(), self.pcilt, self.out_channels, self.kernel_size, 
            self.stride, self.padding, self.activation_bits
        )
        
        out = out + self.bias.view(1, -1, 1, 1)
        return out

# The rest of your PCILTCNN class remains the same