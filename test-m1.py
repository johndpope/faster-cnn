import torch
from pcilt_conv_metal import PCILTConv2DMetal
import numpy as np
import torch.nn as nn
import time
from torch.profiler import profile, record_function, ProfilerActivity
import traceback

def debug_print(message):
    print(f"DEBUG: {message}")

# Create a PyTorch wrapper for the Metal implementation
class PCILTConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation_bits=8, weight_bits=8):
        super().__init__()
        self.metal_conv = PCILTConv2DMetal(in_channels, out_channels, kernel_size, stride, padding, activation_bits, weight_bits)

    def forward(self, x):
        debug_print(f"PCILTConv2D forward - Input shape: {x.shape}")
        input_array = x.detach().cpu().numpy()
        output_array = self.metal_conv.forward(input_array)
        debug_print(f"PCILTConv2D forward - Output shape: {output_array.shape}")
        return torch.from_numpy(output_array).to(x.device)

def profile_conv(conv_layer, input_tensor, num_iterations=100):
    try:
        debug_print(f"Profiling {conv_layer.__class__.__name__}...")
        
        debug_print("Warm-up phase...")
        for _ in range(10):
            _ = conv_layer(input_tensor)
        
        debug_print(f"Speed profiling ({num_iterations} iterations)...")
        start_time = time.time()
        for _ in range(num_iterations):
            _ = conv_layer(input_tensor)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        
        debug_print(f"Profiling completed for {conv_layer.__class__.__name__}")
        return avg_time
    except Exception as e:
        print(f"Error in profile_conv: {str(e)}")
        return None

def compare_conv_layers(in_channels, out_channels, kernel_size, input_size, activation_bits=8, weight_bits=8):
    try:
        print(f"\nComparing Conv2d and PCILTConv2d:")
        print(f"Parameters: in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, input_size={input_size}")
        print(f"PCILT parameters: activation_bits={activation_bits}, weight_bits={weight_bits}")
        
        # Create layers
        print("\nCreating Conv2d layer...")
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        print("Creating PCILTConv2d layer...")
        pcilt_conv = PCILTConv2D(in_channels, out_channels, kernel_size, padding=kernel_size//2, 
                                 activation_bits=activation_bits, weight_bits=weight_bits)

        # Create input tensor
        print(f"\nCreating input tensor with shape (1, {in_channels}, {input_size}, {input_size})...")
        x = torch.randn(1, in_channels, input_size, input_size)
        
        # Profile Conv2d
        print("\nProfiling Conv2d...")
        conv_time = profile_conv(conv, x)
        
        # Profile PCILTConv2d
        print("\nProfiling PCILTConv2d...")
        pcilt_time = profile_conv(pcilt_conv, x)
        
        if conv_time is not None and pcilt_time is not None:
            print("\nResults:")
            print(f"Conv2d - Avg time: {conv_time*1000:.2f} ms")
            print(f"PCILTConv2d - Avg time: {pcilt_time*1000:.2f} ms")
            print(f"Speed difference: {conv_time/pcilt_time:.2f}x {'faster' if conv_time > pcilt_time else 'slower'} than Conv2d")

        print("\nDetailed profiling using torch.profiler...")
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("Conv2d"):
                _ = conv(x)
            with record_function("PCILTConv2d"):
                _ = pcilt_conv(x)

        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    except Exception as e:
        print(f"Error in compare_conv_layers: {str(e)}")

def test_metal_directly():
    try:
        debug_print("Testing PCILTConv2DMetal directly")
        metal_conv = PCILTConv2DMetal(64, 128, 3, stride=1, padding=1, activation_bits=4, weight_bits=4)
        debug_print("PCILTConv2DMetal instance created successfully")
        input_array = np.random.rand(1, 64, 224, 224).astype(np.float32)
        debug_print("Input array created")
        output_array = metal_conv.forward(input_array)
        debug_print(f"Direct Metal test - Output shape: {output_array.shape}")
    except Exception as e:
        print(f"Error in direct Metal test: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == '__main__':
    print("Starting main execution...")
    try:
        test_metal_directly()
        # Example usage
        compare_conv_layers(in_channels=64, out_channels=128, kernel_size=3, input_size=224, activation_bits=4, weight_bits=4)
    except Exception as e:
        print(f"An error occurred in main execution: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
    print("Main execution completed.")

