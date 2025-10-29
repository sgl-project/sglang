"""
Standalone build for MLA RoPE FP8 Fusion kernel
"""
from setuptools import setup
import os
import sys

# Delay torch import until build time
def get_cuda_arch():
    try:
        import torch
        cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if cuda_arch_list is None:
            # Auto-detect
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                cuda_arch_list = f"{capability[0]}.{capability[1]}"
            else:
                # Default to common architectures
                cuda_arch_list = "8.0;9.0;10.0"
        print(f"Building for CUDA architectures: {cuda_arch_list}")
        return cuda_arch_list
    except Exception as e:
        print(f"Warning: Could not detect CUDA arch, using defaults: {e}")
        return "8.0;9.0;10.0"

def get_extensions():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    
    cuda_arch_list = get_cuda_arch()

    return [
        CUDAExtension(
            name='mla_fusion_kernel',
            sources=[
                '../sgl-kernel/csrc/elementwise/mla_rope_fp8_kv_fused.cu',
            ],
            include_dirs=[
                '../sgl-kernel/include',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                ] + [f'-gencode=arch=compute_{arch.replace(".", "")},code=sm_{arch.replace(".", "")}' 
                     for arch in cuda_arch_list.split(';')],
            },
        )
    ]

if __name__ == '__main__':
    from torch.utils.cpp_extension import BuildExtension
    
    setup(
        name='mla_fusion_kernel',
        ext_modules=get_extensions(),
        cmdclass={
            'build_ext': BuildExtension
        },
        python_requires='>=3.8',
    )

