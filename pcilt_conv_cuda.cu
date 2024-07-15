// pcilt_conv_cuda.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void pcilt_conv_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ pcilt,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int activation_bits) {

    const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    const int n = blockIdx.x;
    const int c = blockIdx.y;
    const int h = blockIdx.z / out_width;
    const int w = blockIdx.z % out_width;

    if (n >= batch_size || c >= out_channels || h >= out_height || w >= out_width) return;

    scalar_t sum = 0;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int ih = h * stride + kh - padding;
                const int iw = w * stride + kw - padding;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    const int input_idx = n * (in_channels * height * width) +
                                          ic * (height * width) +
                                          ih * width + iw;
                    const int pcilt_idx = c * (in_channels * kernel_size * kernel_size * (1 << activation_bits)) +
                                          ic * (kernel_size * kernel_size * (1 << activation_bits)) +
                                          kh * (kernel_size * (1 << activation_bits)) +
                                          kw * (1 << activation_bits) +
                                          static_cast<int>(input[input_idx]);
                    sum += pcilt[pcilt_idx];
                }
            }
        }
    }

    const int output_idx = n * (out_channels * out_height * out_width) +
                           c * (out_height * out_width) +
                           h * out_width + w;
    output[output_idx] = sum;
}

torch::Tensor pcilt_conv_cuda(
    torch::Tensor input,
    torch::Tensor pcilt,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int activation_bits) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);

    const auto out_height = (height + 2 * padding - kernel_size) / stride + 1;
    const auto out_width = (width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const dim3 threads(32, 32);
    const dim3 blocks((batch_size + threads.x - 1) / threads.x,
                      (out_channels + threads.y - 1) / threads.y,
                      out_height * out_width);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "pcilt_conv_cuda", ([&] {
        pcilt_conv_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            pcilt.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_size,
            stride,
            padding,
            activation_bits
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pcilt_conv_cuda", &pcilt_conv_cuda, "PCILT Convolution CUDA");
}