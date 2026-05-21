"""Fused-add RMSNorm with HF semantics (cast-before-multiply)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.rmsnorm_hf import is_supported_rmsnorm_hf_hidden_size
from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_CTA_BLOCK_SIZE = 512


def is_supported_fused_add_rmsnorm_hf_hidden_size(hidden_size: int) -> bool:
    return is_supported_rmsnorm_hf_hidden_size(hidden_size)


@cache_once
def _jit_fused_add_rmsnorm_hf_module(
    hidden_size: int, dtype: torch.dtype, has_post_residual: bool
) -> "Module":
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), has_post_residual, dtype)
    kernel_cls = (
        "HFFusedAddRMSNormWarpKernel"
        if hidden_size < _CTA_BLOCK_SIZE
        else "HFFusedAddRMSNormKernel"
    )
    return load_jit(
        "fused_add_rmsnorm_hf",
        *args,
        cuda_files=["elementwise/fused_add_rmsnorm_hf.cuh"],
        cuda_wrappers=[("fused_add_rmsnorm_hf", f"{kernel_cls}<{args}>::run")],
    )


def fused_add_rmsnorm_hf(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    post_residual: Optional[torch.Tensor] = None,
) -> None:
    """In-place: ``input`` <- weight * cast_dtype(rsqrt(mean(s^2) + eps) * s),
    ``residual`` <- s cast to dtype, where s = input + residual (+ post_residual).
    """
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"fused_add_rmsnorm_hf: input must be fp16 or bf16, got {input.dtype}"
        )
    if input.dim() != 2:
        raise RuntimeError(
            f"fused_add_rmsnorm_hf: input must be 2D, got {input.dim()}D"
        )
    if input.shape != residual.shape:
        raise RuntimeError(
            f"fused_add_rmsnorm_hf: input shape {tuple(input.shape)} != "
            f"residual shape {tuple(residual.shape)}"
        )
    if input.dtype != residual.dtype:
        raise RuntimeError(
            f"fused_add_rmsnorm_hf: input dtype {input.dtype} != "
            f"residual dtype {residual.dtype}"
        )
    if post_residual is not None and (
        post_residual.shape != input.shape or post_residual.dtype != input.dtype
    ):
        raise RuntimeError(
            f"fused_add_rmsnorm_hf: post_residual shape/dtype mismatch "
            f"({tuple(post_residual.shape)}, {post_residual.dtype}) vs "
            f"input ({tuple(input.shape)}, {input.dtype})"
        )
    hidden_size = input.size(-1)
    if not is_supported_fused_add_rmsnorm_hf_hidden_size(hidden_size):
        raise RuntimeError(
            f"fused_add_rmsnorm_hf: unsupported hidden_size={hidden_size}"
        )
    if input.numel() == 0:
        return
    module = _jit_fused_add_rmsnorm_hf_module(
        hidden_size, input.dtype, post_residual is not None
    )
    module.fused_add_rmsnorm_hf(input, residual, post_residual, weight, eps)
