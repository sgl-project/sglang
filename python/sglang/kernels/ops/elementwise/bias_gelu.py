from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_bias_gelu_tanh_module(dtype: torch.dtype) -> Module:
    if dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"bias_gelu_tanh does not support {dtype}")
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "bias_gelu_tanh",
        *args,
        cuda_files=["elementwise/bias_gelu.cuh"],
        cuda_wrappers=[("bias_gelu_tanh", f"bias_gelu_tanh<{args}>")],
    )


@register_custom_op(mutates_args=["output"])
def _bias_gelu_tanh(
    input: torch.Tensor, bias: torch.Tensor, output: torch.Tensor
) -> None:
    input_2d = input.view(-1, input.shape[-1])
    output_2d = output.view_as(input_2d)
    module = _jit_bias_gelu_tanh_module(input.dtype)
    module.bias_gelu_tanh(input_2d, bias, output_2d)


def bias_gelu_tanh(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Add a row-wise bias and apply approximate GELU."""
    output = torch.empty_like(input)
    _bias_gelu_tanh(input, bias, output)
    return output
