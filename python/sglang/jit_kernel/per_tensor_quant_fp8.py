from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING

import flashinfer
import torch
from torch.utils.cpp_extension import CUDA_HOME

from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_per_tensor_quant_fp8_module(is_static: bool) -> Module:
    flashinfer_include = os.path.join(os.path.dirname(flashinfer.__file__), "data", "include")
    cub_include = os.path.join(CUDA_HOME, "include")
    torch_include = os.path.join(os.path.dirname(torch.__file__), "include")
    
    kernel_name = "per_tensor_quant_fp8_static" if is_static else "per_tensor_quant_fp8_dynamic"
    
    return load_jit(
        f"per_tensor_quant_fp8_{'static' if is_static else 'dynamic'}",
        cuda_files=["gemm/per_tensor_quant_fp8.cuh"],
        cuda_wrappers=[("per_tensor_quant_fp8", kernel_name)],
        extra_include_paths=[flashinfer_include, cub_include, torch_include],
    )


def per_tensor_quant_fp8(
    input: torch.Tensor, output_q: torch.Tensor, output_s: torch.Tensor, is_static: bool = False
) -> None:
    """
    Per-tensor quantization to FP8 format.
    
    Args:
        input: Input tensor to quantize (float, half, or bfloat16)
        output_q: Output quantized tensor (fp8_e4m3)
        output_s: Output scale tensor (float scalar)
        is_static: If True, assumes scale is pre-computed and skips absmax computation
    """
    module = _jit_per_tensor_quant_fp8_module(is_static)
    module.per_tensor_quant_fp8(input, output_q, output_s)
