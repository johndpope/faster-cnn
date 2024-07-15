# faster-cnn
# Faster Convolution Inference Through Using Pre-Calculated Lookup Tables
https://arxiv.org/pdf/2104.01681



```shell
python setup.py install
```


```shell
Conv2d - Avg time: 0.33 ms, Memory usage: 74.06 MB
PCILTConv2d - Avg time: 275.10 ms, Memory usage: 115.31 MB
Speed difference: 0.00x
Memory difference: 0.64x
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                            PCILTConv2d         0.00%       0.000us         0.00%       0.000us       0.000us     281.103ms        49.74%     281.103ms     281.103ms             1
void pcilt_conv_cuda_kernel<float>(float const*, flo...         0.00%       0.000us         0.00%       0.000us       0.000us     279.927ms        49.53%     279.927ms     279.927ms             1
                                               aten::to         0.42%       1.185ms         1.24%       3.520ms     704.089us       0.000us         0.00%       1.864ms     372.753us             5
                                         aten::_to_copy         0.11%     314.245us         0.82%       2.336ms     583.978us       0.000us         0.00%       1.864ms     465.941us             4
                                            aten::copy_         0.01%      30.722us         0.70%       1.994ms     498.513us       1.864ms         0.33%       1.864ms     465.941us             4
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us       1.768ms         0.31%       1.768ms     883.868us             2
                                                 Conv2d         0.00%       0.000us         0.00%       0.000us       0.000us       1.757ms         0.31%       1.757ms       1.757ms             1
                                                 Conv2d         0.16%     451.266us         1.20%       3.415ms       3.415ms       0.000us         0.00%       1.235ms       1.235ms             1
                                            PCILTConv2d         0.05%     129.235us         0.46%       1.322ms       1.322ms       0.000us         0.00%       1.122ms       1.122ms             1
                                           aten::conv2d         0.04%     128.053us         0.15%     417.854us     417.854us       0.000us         0.00%     307.096us     307.096us             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 284.599ms
Self CUDA time total: 565.145ms
```