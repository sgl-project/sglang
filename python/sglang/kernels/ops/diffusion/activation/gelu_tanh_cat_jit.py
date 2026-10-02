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


def can_use_fused_gelu_tanh_cat(attn: torch.Tensor, mlp: torch.Tensor) -> bool:
    return (
        not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and torch.version.hip is None
        and attn.is_cuda
        and attn.dtype == mlp.dtype == torch.bfloat16
        and attn.device == mlp.device
        and attn.ndim >= 2
        and attn.shape[:-1] == mlp.shape[:-1]
        and attn.shape[-1] > 0
        and mlp.shape[-1] > 0
        and attn.shape[-1] % 8 == mlp.shape[-1] % 8 == 0
        and attn.is_contiguous()
        and mlp.is_contiguous()
        and attn.numel() > 0
        and (attn.numel() + mlp.numel()) // 8 <= 2**31 - 1
        and attn.data_ptr() % 16 == mlp.data_ptr() % 16 == 0
    )


@register_custom_op(mutates_args=["output"])
def _gelu_tanh_cat(attn: torch.Tensor, mlp: torch.Tensor, output: torch.Tensor) -> None:
    _module().gelu_tanh_cat(
        attn.view(-1, attn.shape[-1]),
        mlp.view(-1, mlp.shape[-1]),
        output.view(-1, output.shape[-1]),
    )


def fused_gelu_tanh_cat(attn: torch.Tensor, mlp: torch.Tensor) -> torch.Tensor:
    output = torch.empty(
        (*attn.shape[:-1], attn.shape[-1] + mlp.shape[-1]),
        device=attn.device,
        dtype=attn.dtype,
    )
    _gelu_tanh_cat(attn, mlp, output)
    return output
