from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_per_tensor_quant_fp8_module(is_static: bool, dtype: torch.dtype) -> Module:
    args = make_cpp_args(is_static, dtype)
    return load_jit(
        "per_tensor_quant_fp8",
        *args,
        cuda_files=["gemm/per_tensor_quant_fp8.cuh"],
        cuda_wrappers=[("per_tensor_quant_fp8", f"per_tensor_quant_fp8<{args}>")],
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
    module = _jit_per_tensor_quant_fp8_module(is_static, input.dtype)
    module.per_tensor_quant_fp8(input.view(-1), output_q.view(-1), output_s.view(-1))
