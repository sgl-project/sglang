from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_qknorm_module(head_dim: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(head_dim, is_arch_support_pdl(), dtype)
    return load_jit(
        "qknorm",
        *args,
        cuda_files=["elementwise/qknorm.cuh"],
        cuda_wrappers=[("qknorm", f"QKNormKernel<{args}>::run")],
    )


@cache_once
def _jit_rmsnorm_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "rmsnorm",
        *args,
        cuda_files=["elementwise/rmsnorm.cuh"],
        cuda_wrappers=[("rmsnorm", f"RMSNormKernel<{args}>::run")],
    )


@cache_once
def _jit_fused_add_rmsnorm_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "fused_add_rmsnorm",
        *args,
        cuda_files=["elementwise/fused_add_rmsnorm.cuh"],
        cuda_wrappers=[("fused_add_rmsnorm", f"FusedAddRMSNormKernel<{args}>::run")],
    )


@cache_once
def _jit_qknorm_across_heads_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "qknorm_across_heads",
        *args,
        cuda_files=["elementwise/qknorm_across_heads.cuh"],
        cuda_wrappers=[
            ("qknorm_across_heads", f"QKNormAcrossHeadsKernel<{args}>::run")
        ],
    )


@cache_once
def can_use_fused_inplace_qknorm(head_dim: int, dtype: torch.dtype) -> bool:
    logger = logging.getLogger(__name__)
    if head_dim not in [64, 128, 256, 512, 1024]:
        logger.warning(f"Unsupported head_dim={head_dim} for JIT QK-Norm kernel")
        return False
    try:
        _jit_qknorm_module(head_dim, dtype)
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT QK-Norm kernel: {e}")
        return False


def fused_inplace_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    head_dim: int = 0,
) -> None:
    head_dim = head_dim or q.size(-1)
    module = _jit_qknorm_module(head_dim, q.dtype)
    module.qknorm(q, k, q_weight, k_weight, eps)


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> None:
    output = output if output is not None else input
    hidden_size = input.size(-1)
    module = _jit_rmsnorm_module(hidden_size, input.dtype)
    module.rmsnorm(input, weight, output, eps)


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    module = _jit_fused_add_rmsnorm_module(input.dtype)
    module.fused_add_rmsnorm(input, residual, weight, eps)


def fused_inplace_qknorm_across_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """
    Fused inplace QK normalization across all heads.

    Args:
        q: Query tensor of shape [batch_size, num_heads * head_dim]
        k: Key tensor of shape [batch_size, num_heads * head_dim]
        q_weight: Query weight tensor of shape [num_heads * head_dim]
        k_weight: Key weight tensor of shape [num_heads * head_dim]
        eps: Epsilon for numerical stability
    """
    module = _jit_qknorm_across_heads_module(q.dtype)
    module.qknorm_across_heads(q, k, q_weight, k_weight, eps)
