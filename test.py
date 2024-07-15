import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import random
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

def profile_conv(conv_layer, input_tensor, num_iterations=100):
    try:
        conv_layer.cuda()
        input_tensor = input_tensor.cuda()
        
        # Warm-up
        for _ in range(10):
            _ = conv_layer(input_tensor)
        
        torch.cuda.synchronize()
        
        # Speed profiling
        start_time = time.time()
        for _ in range(num_iterations):
            _ = conv_layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        
        # Memory profiling
        torch.cuda.reset_peak_memory_stats()
        _ = conv_layer(input_tensor)
        torch.cuda.synchronize()
        memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
        
        return avg_time, memory_usage
    except Exception as e:
        print(f"Error in profile_conv: {str(e)}")
        return None, None

def compare_conv_layers(in_channels, out_channels, kernel_size, input_size, activation_bits=8, weight_bits=8):
    try:
        # Create layers
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        pcilt_conv = PCILTConv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, 
                                 activation_bits=activation_bits, weight_bits=weight_bits)

        # Create input tensor
        x = torch.randn(1, in_channels, input_size, input_size)
        
        # Profile Conv2d
        conv_time, conv_memory = profile_conv(conv, x)
        
        # Profile PCILTConv2d
        pcilt_time, pcilt_memory = profile_conv(pcilt_conv, x)
        
        if conv_time is not None and pcilt_time is not None:
            print(f"Conv2d - Avg time: {conv_time*1000:.2f} ms, Memory usage: {conv_memory:.2f} MB")
            print(f"PCILTConv2d - Avg time: {pcilt_time*1000:.2f} ms, Memory usage: {pcilt_memory:.2f} MB")
            print(f"Speed difference: {conv_time/pcilt_time:.2f}x")
            print(f"Memory difference: {conv_memory/pcilt_memory:.2f}x")

        # Detailed profiling using torch.profiler
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("Conv2d"):
                _ = conv(x.cuda())
            with record_function("PCILTConv2d"):
                _ = pcilt_conv(x.cuda())

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    except Exception as e:
        print(f"Error in compare_conv_layers: {str(e)}")

if __name__ == '__main__':
    # Example usage
    compare_conv_layers(in_channels=64, out_channels=128, kernel_size=3, input_size=224, activation_bits=4, weight_bits=4)