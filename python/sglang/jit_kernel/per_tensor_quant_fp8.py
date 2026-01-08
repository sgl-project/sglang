from __future__ import annotations

import os
from typing import TYPE_CHECKING

import flashinfer
import torch
from torch.utils.cpp_extension import CUDA_HOME

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_per_tensor_quant_fp8_module(is_static: bool) -> Module:
    args = make_cpp_args(is_static)

    flashinfer_include = os.path.join(
        os.path.dirname(flashinfer.__file__), "data", "include"
    )
    cub_include = os.path.join(CUDA_HOME, "include")

    return load_jit(
        "per_tensor_quant_fp8",
        *args,
        cuda_files=["gemm/per_tensor_quant_fp8.cuh"],
        cuda_wrappers=[("per_tensor_quant_fp8", f"per_tensor_quant_fp8<{args}>")],
        extra_include_paths=[flashinfer_include, cub_include],
    )


@register_custom_op(
    op_name="per_tensor_quant_fp8",
    mutates_args=["output_q", "output_s"],
)
def per_tensor_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    is_static: bool = False,
) -> None:
    """
    Per-tensor quantization to FP8 format.

    Args:
        input: Input tensor to quantize (float, half, or bfloat16)
        output_q: Output quantized tensor (fp8_e4m3)
        output_s: Output scale tensor (float scalar or 1D tensor with 1 element)
        is_static: If True, assumes scale is pre-computed and skips absmax computation
    """
    # Ensure output_s has shape [1] instead of being a 0D scalar
    # The JIT kernel expects a 1D tensor
    if output_s.ndim == 0:
        output_s = output_s.reshape(1)

    module = _jit_per_tensor_quant_fp8_module(is_static)
    module.per_tensor_quant_fp8(input, output_q, output_s)
