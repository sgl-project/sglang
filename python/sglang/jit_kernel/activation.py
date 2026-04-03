from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_activation_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "activation",
        *args,
        cuda_files=["elementwise/activation.cuh"],
        extra_cuda_cflags=["--use_fast_math"],
        cuda_wrappers=[
            ("run_activation", f"ActivationKernel<{args}>::run_activation"),
        ],
    )


SUPPORTED_ACTIVATIONS = {"silu", "gelu", "gelu_tanh"}


@register_custom_op(mutates_args=["out"], out_shape="input")
def run_activation(
    op_name: str, input: torch.Tensor, out: Optional[torch.Tensor]
) -> torch.Tensor:
    assert op_name in SUPPORTED_ACTIVATIONS, f"Unsupported activation: {op_name}"

    if out is None:
        out = input.new_empty(*input.shape[:-1], input.shape[-1] // 2)
    module = _jit_activation_module(input.dtype)
    input_2d = input.view(-1, input.shape[-1])
    out_2d = out.view(-1, out.shape[-1])
    module.run_activation(input_2d, out_2d, op_name)
    return out


def silu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return run_activation("silu", input, out)


def gelu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return run_activation("gelu", input, out)


def gelu_tanh_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return run_activation("gelu_tanh", input, out)
