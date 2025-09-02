"""
Setup script for building MXFP4 grouped GEMM kernels.
Alternative to CMake for easier Python integration.
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

def get_cuda_arch_list():
    """Get CUDA architectures to compile for, respecting TORCH_CUDA_ARCH_LIST."""
    # Check environment variable first
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "12.0")
    return arch_list.split(";")

setup(
    name='mxfp4_kernels',
    ext_modules=[
        CUDAExtension(
            name='_mxfp4_kernels',
            sources=[
                'mxfp4_grouped.cpp',
                'mxfp4_grouped.cu',
            ],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-std=c++17',
                    '-fPIC',
                    '-Wall',
                ],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '-lineinfo',
                    '--expt-extended-lambda',
                    '--expt-relaxed-constexpr',
                ] + [f'-gencode=arch=compute_{a.replace(".", "")},code=sm_{a.replace(".", "")}' 
                     for a in get_cuda_arch_list()],
            },
            extra_link_args=['-Wl,--no-as-needed'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    install_requires=[
        'torch',
    ],
)