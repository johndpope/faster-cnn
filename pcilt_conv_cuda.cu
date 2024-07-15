#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void pcilt_conv_cuda_kernel_optimized(
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

    extern __shared__ char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);

    const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    const int n = blockIdx.x;
    const int c_out = blockIdx.y;
    const int h_out = (blockIdx.z / ((out_width + blockDim.x - 1) / blockDim.x)) * blockDim.y + threadIdx.y;
    const int w_out = (blockIdx.z % ((out_width + blockDim.x - 1) / blockDim.x)) * blockDim.x + threadIdx.x;

    if (h_out >= out_height || w_out >= out_width) return;

    scalar_t sum = 0;

    const int h_in = h_out * stride - padding;
    const int w_in = w_out * stride - padding;

    // Load input patch into shared memory
    for (int i = tid; i < (kernel_size * kernel_size * in_channels); i += block_size) {
        int ic = i / (kernel_size * kernel_size);
        int kh = (i % (kernel_size * kernel_size)) / kernel_size;
        int kw = i % kernel_size;

        int h = h_in + kh;
        int w = w_in + kw;

        if (h >= 0 && h < height && w >= 0 && w < width) {
            shared_input[i] = input[(n * in_channels + ic) * height * width + h * width + w];
        } else {
            shared_input[i] = 0;
        }
    }

    __syncthreads();

    // Compute convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int input_idx = (ic * kernel_size + kh) * kernel_size + kw;
                int pcilt_idx = ((c_out * in_channels + ic) * kernel_size + kh) * kernel_size * (1 << activation_bits) + kw * (1 << activation_bits) + static_cast<int>(shared_input[input_idx]);
                sum += pcilt[pcilt_idx];
            }
        }
    }

    if (h_out < out_height && w_out < out_width) {
        output[(n * out_channels + c_out) * out_height * out_width + h_out * out_width + w_out] = sum;
    }
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

    const dim3 threads(16, 16);
    const dim3 blocks(batch_size,
                      out_channels,
                      ((out_height + threads.y - 1) / threads.y) * ((out_width + threads.x - 1) / threads.x));

    const int shared_memory_size = in_channels * kernel_size * kernel_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "pcilt_conv_cuda", ([&] {
        pcilt_conv_cuda_kernel_optimized<scalar_t><<<blocks, threads, shared_memory_size>>>(
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