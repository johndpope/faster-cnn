# faster-cnn
# Faster Convolution Inference Through Using Pre-Calculated Lookup Tables
https://arxiv.org/pdf/2104.01681


M1 - optimization
```shell
pip install pyobjc-core pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders
python test-m1.py
```


```shell
python setup.py install
```

i don't understand why this is actually slower 

```shell
Results:
Conv2d - Avg time: 0.33 ms, Memory usage: 74.06 MB
PCILTConv2d - Avg time: 4.77 ms, Memory usage: 115.31 MB
Speed difference: 0.07x slower than Conv2d
Memory difference: 0.64x less than Conv2d

Detailed profiling using torch.profiler...
PCILTConv2d forward pass - Input shape: torch.Size([1, 64, 224, 224])
Quantized input shape: torch.Size([1, 64, 224, 224])
PCILTConv2d output shape: torch.Size([1, 128, 224, 224])
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                            PCILTConv2d         0.00%       0.000us         0.00%       0.000us       0.000us       5.648ms        40.76%       5.648ms       5.648ms             1
void pcilt_conv_cuda_kernel_optimized<float>(float c...         0.00%       0.000us         0.00%       0.000us       0.000us       4.454ms        32.14%       4.454ms       4.454ms             1
                                               aten::to        12.59%       1.034ms        37.71%       3.095ms     619.078us       0.000us         0.00%       1.851ms     370.189us             5
                                         aten::_to_copy         0.36%      29.939us        25.12%       2.062ms     515.463us       0.000us         0.00%       1.851ms     462.736us             4
                                            aten::copy_         0.34%      27.543us        24.43%       2.006ms     501.393us       1.851ms        13.36%       1.851ms     462.736us             4
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us       1.754ms        12.65%       1.754ms     876.751us             2
                                                 Conv2d         0.00%       0.000us         0.00%       0.000us       0.000us       1.393ms        10.05%       1.393ms       1.393ms             1
                                                 Conv2d         2.35%     192.715us        29.85%       2.450ms       2.450ms       0.000us         0.00%       1.245ms       1.245ms             1
                                            PCILTConv2d         1.97%     161.865us        16.68%       1.369ms       1.369ms       0.000us         0.00%       1.116ms       1.116ms             1
                                           aten::conv2d         0.07%       5.504us         1.97%     162.025us     162.025us       0.000us         0.00%     322.649us     322.649us             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 8.209ms
Self CUDA time total: 13.857ms

Main execution completed.
```