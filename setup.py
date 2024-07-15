from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pcilt_conv_cuda',
    ext_modules=[
        CUDAExtension('pcilt_conv_cuda', [
            'pcilt_conv_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })