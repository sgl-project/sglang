"""Concatenate attention with tanh GELU while retaining the BF16 rounding boundary.

The inputs are contiguous BF16 tensors with equal leading dimensions and
16-byte aligned rows. The result matches ``cat((attn, gelu(mlp)), dim=-1)``.
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _module():
    args = make_cpp_args(torch.bfloat16)
    return load_jit(
        "gelu_tanh_cat",
        *args,
        cuda_files=["diffusion/gelu_tanh_cat.cuh"],
        cuda_wrappers=[("gelu_tanh_cat", f"gelu_tanh_cat<{args}>")],
        extra_cuda_cflags=["-lineinfo"],
    )


@register_custom_op(mutates_args=["output"])
def _gelu_tanh_cat(attn: torch.Tensor, mlp: torch.Tensor, output: torch.Tensor) -> None:
    _module().gelu_tanh_cat(attn, mlp, output)


def fused_gelu_tanh_cat(attn: torch.Tensor, mlp: torch.Tensor) -> torch.Tensor:
    output = torch.empty(
        (*attn.shape[:-1], attn.shape[-1] + mlp.shape[-1]),
        device=attn.device,
        dtype=attn.dtype,
    )
    _gelu_tanh_cat(attn, mlp, output)
    return output
