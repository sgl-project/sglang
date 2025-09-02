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
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    if arch_list_env:
        # Parse various formats: "12.0", "12.0;11.0", "12.0 11.0", "120+PTX"
        arch_list_env = arch_list_env.replace(";", " ").replace(",", " ")
        archs = []
        for arch in arch_list_env.split():
            # Remove +PTX suffix if present
            arch = arch.split("+")[0]
            # Normalize format (12.0 -> 12.0, 120 -> 12.0)
            if "." not in arch:
                if len(arch) == 3:
                    arch = f"{arch[0:2]}.{arch[2]}"
                elif len(arch) == 2:
                    arch = f"{arch}.0"
            archs.append(arch)
        return archs if archs else ["12.0"]  # Default to SM120
    else:
        # Default to SM120 for RTX 5090
        return ["12.0"]

setup(
    name='mxfp4_kernels',
    ext_modules=[
        CUDAExtension(
            name='_mxfp4_kernels',
            sources=[
                'mxfp4_grouped.cpp',
                'mxfp4_grouped_impl.cu',
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