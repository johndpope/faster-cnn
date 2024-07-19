import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import random
import pcilt_conv_cuda

print("Importing necessary libraries...")


import torch
import torch.nn as nn
import torch.nn.functional as F

class QATPCILTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation_bits=8, weight_bits=8):
        super(QATPCILTConv2d, self).__init__()
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
        
        # Initialize scaling factors for quantization
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.activation_scale = nn.Parameter(torch.ones(1))

    def quantize_weight(self, weight):
        weight_q = torch.round(weight * self.weight_scale * (2**self.weight_bits - 1))
        weight_q = torch.clamp(weight_q, 0, 2**self.weight_bits - 1)
        return weight_q / (self.weight_scale * (2**self.weight_bits - 1))

    def quantize_activation(self, x):
        x_q = torch.round(x * self.activation_scale * (2**self.activation_bits - 1))
        x_q = torch.clamp(x_q, 0, 2**self.activation_bits - 1)
        return x_q / (self.activation_scale * (2**self.activation_bits - 1))

    def create_pcilt(self):
        weight_q = self.quantize_weight(self.weight)
        activation_range = 2 ** self.activation_bits
        
        pcilt = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, activation_range)
        for i in range(activation_range):
            pcilt[:, :, :, :, i] = weight_q * (i / (2**self.activation_bits - 1))
        
        return pcilt

    def forward(self, x):
        if self.training:
            # During training, use straight-through estimator
            x_q = self.quantize_activation(x)
            weight_q = self.quantize_weight(self.weight)
            out = F.conv2d(x_q, weight_q, self.bias, self.stride, self.padding)
            out = out + (out.detach() - out)  # Straight-through estimator
        else:
            # During inference, use PCILT
            pcilt = self.create_pcilt()
            x_q = (x * (2**self.activation_bits - 1)).round().clamp(0, 2**self.activation_bits - 1).long()
            out = F.conv2d(x_q.float(), pcilt[:, :, :, :, x_q].sum(dim=-1), self.bias, self.stride, self.padding)
        
        return out

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, ' \
               f'stride={self.stride}, padding={self.padding}, activation_bits={self.activation_bits}, ' \
               f'weight_bits={self.weight_bits}'
    
class WinogradPCILTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, activation_bits=8, weight_bits=8):
        super(WinogradPCILTConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits
        
        # Winograd transform matrices
        self.G = torch.tensor([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]])
        self.B = torch.tensor([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]])
        self.A = torch.tensor([[1, 0], [1, 1], [1, -1], [0, -1]])
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize Winograd-transformed PCILTs
        self.winograd_pcilt = self._create_winograd_pcilt()

    def _create_winograd_pcilt(self):
        activation_range = 2 ** self.activation_bits
        weight_range = 2 ** self.weight_bits
        
        # Transform weights using Winograd
        transformed_weights = torch.einsum('ij,klmj,no->klmino', self.G, self.weight, self.G.t())
        
        # Create PCILTs for transformed weights
        pcilt = torch.zeros(self.out_channels, self.in_channels, 4, 4, activation_range, dtype=torch.float32)
        
        quantized_weights = self._quantize_weights(transformed_weights, weight_range)
        
        for i in range(activation_range):
            pcilt[:, :, :, :, i] = quantized_weights * i
        
        return nn.Parameter(pcilt, requires_grad=False)

    def _quantize_weights(self, weights, weight_range):
        return (weights * (weight_range - 1)).round().clamp(0, weight_range - 1)

    def forward(self, x):
        # Assume input is already padded appropriately
        batch, channels, height, width = x.shape
        output_height, output_width = height - 2, width - 2
        
        # Quantize input
        x_quant = (x * (2**self.activation_bits - 1)).round().clamp(0, 2**self.activation_bits - 1).long()
        
        # Transform input patches
        x_transformed = F.unfold(x_quant.float(), 4, stride=2).view(batch, channels, 16, -1)
        x_transformed = torch.einsum('ij,bckm->bcijm', self.B, x_transformed)
        
        # Perform element-wise multiplication using PCILTs
        out = torch.zeros(batch, self.out_channels, 16, x_transformed.size(-1), device=x.device)
        for i in range(4):
            for j in range(4):
                indices = x_transformed[:, :, i, j, :]
                out[:, :, i*4+j, :] = self.winograd_pcilt[:, :, i, j, indices].sum(dim=1)
        
        # Inverse transform
        out = torch.einsum('ij,bcjk->bcik', self.A, out.view(batch, self.out_channels, 4, 4, -1))
        out = out.view(batch, self.out_channels, output_height, output_width)
        
        return out + self.bias.view(1, -1, 1, 1)
    
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
        
        print(f"Initializing PCILTConv2d: in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}")
        print(f"Activation bits: {activation_bits}, Weight bits: {weight_bits}")
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize PCILTs
        self.pcilt = self._create_pcilt()
        print(f"PCILT shape: {self.pcilt.shape}")

    def _create_pcilt(self):
        print("Creating PCILT...")
        activation_range = 2 ** self.activation_bits
        weight_range = 2 ** self.weight_bits
        
        pcilt = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, activation_range, dtype=torch.float32)
        
        # Quantize weights
        quantized_weights = self._quantize_weights(self.weight, weight_range)
        
        for i in range(activation_range):
            pcilt[:, :, :, :, i] = quantized_weights * i
        print("PCILT creation completed.")
        return nn.Parameter(pcilt, requires_grad=False)

    def _quantize_weights(self, weights, weight_range):
        return (weights * (weight_range - 1)).round().clamp(0, weight_range - 1)

    def forward(self, x):
        print(f"PCILTConv2d forward pass - Input shape: {x.shape}")
        # Quantize input to specified bit depth
        x_quant = (x * (2**self.activation_bits - 1)).round().clamp(0, 2**self.activation_bits - 1).long()
        print(f"Quantized input shape: {x_quant.shape}")
        
        # Use CUDA implementation
        out = pcilt_conv_cuda.pcilt_conv_cuda(
            x_quant.float(), self.pcilt, self.out_channels, self.kernel_size, 
            self.stride, self.padding, self.activation_bits
        )
        
        out = out + self.bias.view(1, -1, 1, 1)
        print(f"PCILTConv2d output shape: {out.shape}")
        return out

def profile_conv(conv_layer, input_tensor, num_iterations=100):
    try:
        print(f"Profiling {conv_layer.__class__.__name__}...")
        conv_layer.cuda()
        input_tensor = input_tensor.cuda()
        
        print("Warm-up phase...")
        for _ in range(10):
            _ = conv_layer(input_tensor)
        
        torch.cuda.synchronize()
        
        print(f"Speed profiling ({num_iterations} iterations)...")
        start_time = time.time()
        for _ in range(num_iterations):
            _ = conv_layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        
        print("Memory profiling...")
        torch.cuda.reset_peak_memory_stats()
        _ = conv_layer(input_tensor)
        torch.cuda.synchronize()
        memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
        
        print(f"Profiling completed for {conv_layer.__class__.__name__}")
        return avg_time, memory_usage
    except Exception as e:
        print(f"Error in profile_conv: {str(e)}")
        return None, None

def compare_conv_layers(in_channels, out_channels, kernel_size, input_size, activation_bits=8, weight_bits=8):
    try:
        print(f"\nComparing Conv2d and PCILTConv2d:")
        print(f"Parameters: in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, input_size={input_size}")
        print(f"PCILT parameters: activation_bits={activation_bits}, weight_bits={weight_bits}")
        
        # Create layers
        print("\nCreating Conv2d layer...")
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        print("Creating PCILTConv2d layer...")
        pcilt_conv = WinogradPCILTConv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, 
                                 activation_bits=activation_bits, weight_bits=weight_bits)

        # Create input tensor
        print(f"\nCreating input tensor with shape (1, {in_channels}, {input_size}, {input_size})...")
        x = torch.randn(1, in_channels, input_size, input_size)
        
        # Profile Conv2d
        print("\nProfiling Conv2d...")
        conv_time, conv_memory = profile_conv(conv, x)
        
        # Profile PCILTConv2d
        print("\nProfiling PCILTConv2d...")
        pcilt_time, pcilt_memory = profile_conv(pcilt_conv, x)
        
        if conv_time is not None and pcilt_time is not None:
            print("\nResults:")
            print(f"Conv2d - Avg time: {conv_time*1000:.2f} ms, Memory usage: {conv_memory:.2f} MB")
            print(f"PCILTConv2d - Avg time: {pcilt_time*1000:.2f} ms, Memory usage: {pcilt_memory:.2f} MB")
            print(f"Speed difference: {conv_time/pcilt_time:.2f}x {'faster' if conv_time > pcilt_time else 'slower'} than Conv2d")
            print(f"Memory difference: {conv_memory/pcilt_memory:.2f}x {'more' if conv_memory > pcilt_memory else 'less'} than Conv2d")

        print("\nDetailed profiling using torch.profiler...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("Conv2d"):
                _ = conv(x.cuda())
            with record_function("PCILTConv2d"):
                _ = pcilt_conv(x.cuda())

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    except Exception as e:
        print(f"Error in compare_conv_layers: {str(e)}")

if __name__ == '__main__':
    print("Starting main execution...")
    # Example usage
    compare_conv_layers(in_channels=64, out_channels=128, kernel_size=3, input_size=224, activation_bits=4, weight_bits=4)
    print("Main execution completed.")