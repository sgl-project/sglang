from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

from sglang.jit_kernel.utils import CPP_DTYPE_MAP as OUTPUT_DTYPE_MAP


@cache_once
def _jit_per_token_group_quant_8bit_module(
    dtype: torch.dtype, output_type: torch.dtype
) -> Module:
    input_args = make_cpp_args(dtype)
    out_cpp = OUTPUT_DTYPE_MAP[output_type]
    return load_jit(
        "per_token_group_quant_8bit",
        cuda_files=["gemm/per_token_group_quant_8bit.cuh"],
        cuda_wrappers=[
            (
                "per_token_group_quant_8bit",
                f"per_token_group_quant_8bit<{input_args}, {out_cpp}>",
            )
        ],
    )


@register_custom_op(
    op_name="per_token_group_quant_8bit",
    mutates_args=["output_q", "output_s"],
)
def _per_token_group_quant_8bit_custom_op(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool = False,
) -> None:
    """
    Per-token-group quantization to 8-bit format.

    Args:
        input: Input tensor to quantize (float, half, or bfloat16).
        output_q: Output quantized tensor (e.g., fp8_e4m3 or int8).
        output_s: Output scale tensor.
        group_size: The size of the group for quantization.
        eps: A small value to avoid division by zero.
        fp8_min: The minimum value of the 8-bit data type.
        fp8_max: The maximum value of the 8-bit data type.
        scale_ue8m0: Whether to use UE8M0 format for scales.
    """
    module = _jit_per_token_group_quant_8bit_module(input.dtype, output_q.dtype)
    module.per_token_group_quant_8bit(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
    )
    return None


def per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _per_token_group_quant_8bit_custom_op(
        input=input,
        output_q=output_q,
        output_s=output_s,
        group_size=group_size,
        eps=eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        scale_ue8m0=scale_ue8m0,
    )
    return output_q, output_s
