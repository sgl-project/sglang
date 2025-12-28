from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
def _jit_norm_module(head_dims: int) -> Module:
    args = make_cpp_args(head_dims, is_arch_support_pdl())
    return load_jit(
        "norm",
        *args,
        cuda_files=["norm.cuh"],
        cuda_wrappers=[("qknorm", f"QKNormKernel<{args}>::run")],
    )


@cache_once
def can_use_fused_inplace_qknorm(head_dim: int) -> bool:
    logger = logging.getLogger(__name__)
    if head_dim not in [64, 128, 256]:
        logger.warning(f"Unsupported head_dim={head_dim} for JIT QK-Norm kernel")
        return False
    try:
        _jit_norm_module(head_dim)
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
    module = _jit_norm_module(head_dim)
    module.qknorm(q, k, q_weight, k_weight, eps)
