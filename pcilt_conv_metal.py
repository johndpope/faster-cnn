# pcilt_conv_metal.py

import numpy as np
from Metal import MTLCreateSystemDefaultDevice, MTLResourceStorageModeShared
from MetalPerformanceShaders import (
    MPSCNNConvolutionDescriptor,
    MPSCNNConvolution
)
import ctypes

def debug_print(message):
    print(f"DEBUG: {message}")

class PCILTConv2DMetal:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation_bits=8, weight_bits=8):
        debug_print("Initializing PCILTConv2DMetal")
        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Failed to create Metal device")
        debug_print("Metal device created successfully")
        
        self.command_queue = self.device.newCommandQueue()
        if self.command_queue is None:
            raise RuntimeError("Failed to create command queue")
        debug_print("Command queue created successfully")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits

        # Create PCILT buffer
        pcilt_size = out_channels * in_channels * kernel_size * kernel_size
        self.pcilt_buffer = self.device.newBufferWithLength_options_(
            pcilt_size * 4,  # 4 bytes per float
            MTLResourceStorageModeShared
        )
        if self.pcilt_buffer is None:
            raise RuntimeError("Failed to create PCILT buffer")
        debug_print("PCILT buffer created successfully")

        debug_print("Initializing PCILT data")
        self._initialize_pcilt()

        debug_print("Creating convolution descriptor")
        self.conv_desc = None
        self._create_conv_descriptor()

        debug_print("Creating convolution")
        self.convolution = None
        self._create_convolution()

        debug_print("Initialization complete")

    def _initialize_pcilt(self):
        debug_print("Generating random PCILT data")
        pcilt_data = np.random.rand(self.pcilt_buffer.length() // 4).astype(np.float32)
        debug_print(f"PCILT data shape: {pcilt_data.shape}")
        buffer = self.pcilt_buffer.contents().as_buffer(self.pcilt_buffer.length())
        np.frombuffer(buffer, dtype=np.float32)[:] = pcilt_data
        debug_print("PCILT data initialized")

    def _create_conv_descriptor(self):
        try:
            self.conv_desc = MPSCNNConvolutionDescriptor.alloc().initWithKernelWidth_kernelHeight_inputFeatureChannels_outputFeatureChannels_(
                self.kernel_size, self.kernel_size, self.in_channels, self.out_channels
            )
            if self.conv_desc is None:
                raise RuntimeError("Failed to create convolution descriptor")
            self.conv_desc.setStrideInPixelsX_(self.stride)
            self.conv_desc.setStrideInPixelsY_(self.stride)
            debug_print("Convolution descriptor created successfully")
        except Exception as e:
            debug_print(f"Error creating convolution descriptor: {str(e)}")
            raise

    def _create_convolution(self):
        try:
            self.convolution = MPSCNNConvolution.alloc().initWithDevice_convolutionDescriptor_kernelWeights_biasTerms_(
                self.device,
                self.conv_desc,
                self.pcilt_buffer,
                None  # No bias terms
            )
            if self.convolution is None:
                raise RuntimeError("Failed to create convolution")
            self.convolution.setPaddingLeft_right_top_bottom_(self.padding, self.padding, self.padding, self.padding)
            debug_print("Convolution created successfully")
        except Exception as e:
            debug_print(f"Error creating convolution: {str(e)}")
            raise

    def forward(self, input_array):
        debug_print(f"Forward pass - Input shape: {input_array.shape}")
        input_texture = self._array_to_texture(input_array)
        if input_texture is None:
            raise RuntimeError("Failed to create input texture")

        output_width = input_array.shape[3]
        output_height = input_array.shape[2]
        debug_print(f"Calculated output dimensions: {output_height}x{output_width}")

        output_desc = self.device.newTextureDescriptorWithPixelFormat_width_height_mipmapped_(
            80,  # MTLPixelFormatRGBA32Float
            output_width, output_height, False
        )
        output_texture = self.device.newTextureWithDescriptor_(output_desc)
        if output_texture is None:
            raise RuntimeError("Failed to create output texture")

        debug_print("Performing convolution")
        command_buffer = self.command_queue.commandBuffer()
        self.convolution.encodeToCommandBuffer_sourceImage_destinationImage_(
            command_buffer, input_texture, output_texture
        )
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        debug_print("Converting output texture to numpy array")
        output_array = self._texture_to_array(output_texture)
        debug_print(f"Output array shape: {output_array.shape}")
        return output_array

    def _array_to_texture(self, array):
        debug_print(f"Converting array to texture - Array shape: {array.shape}")
        texture_desc = self.device.newTextureDescriptorWithPixelFormat_width_height_mipmapped_(
            80,  # MTLPixelFormatRGBA32Float
            array.shape[3], array.shape[2], False
        )
        texture = self.device.newTextureWithDescriptor_(texture_desc)
        if texture is None:
            raise RuntimeError("Failed to create texture from array")
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            (0, 0, 0), 0, array.astype(np.float32).tobytes(),
            array.shape[3] * 4 * 4  # 4 channels, 4 bytes per float
        )
        return texture

    def _texture_to_array(self, texture):
        debug_print(f"Converting texture to array - Texture dimensions: {texture.width()}x{texture.height()}")
        array = np.empty((self.out_channels, texture.height(), texture.width()), dtype=np.float32)
        texture.getBytes_bytesPerRow_fromRegion_mipmapLevel_(
            array.ctypes.data_as(ctypes.c_void_p), texture.width() * 4,
            (0, 0, texture.width(), texture.height()), 0
        )
        return array

# Usage example
if __name__ == "__main__":
    debug_print("Starting main")
    try:
        pcilt_conv = PCILTConv2DMetal(64, 128, 3, stride=1, padding=1, activation_bits=4, weight_bits=4)
        input_array = np.random.rand(1, 64, 224, 224).astype(np.float32)
        output_array = pcilt_conv.forward(input_array)
        print(f"Output shape: {output_array.shape}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    debug_print("Main completed")