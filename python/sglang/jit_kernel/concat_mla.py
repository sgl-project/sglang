from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_concat_mla_k_module() -> Module:
    return load_jit(
        "concat_mla_k",
        cuda_files=["elementwise/concat_mla.cuh"],
        cuda_wrappers=[("concat_mla_k", "ConcatMlaKKernel::run")],
    )


@cache_once
def _jit_concat_mla_absorb_q_module() -> Module:
    return load_jit(
        "concat_mla_absorb_q",
        cuda_files=["elementwise/concat_mla.cuh"],
        cuda_wrappers=[("concat_mla_absorb_q", "ConcatMlaAbsorbQKernel::run")],
    )


@cache_once
def can_use_jit_concat_mla_k() -> bool:
    logger = logging.getLogger(__name__)
    try:
        _jit_concat_mla_k_module()
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT concat_mla_k kernel: {e}")
        return False


@cache_once
def can_use_jit_concat_mla_absorb_q() -> bool:
    logger = logging.getLogger(__name__)
    try:
        _jit_concat_mla_absorb_q_module()
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT concat_mla_absorb_q kernel: {e}")
        return False


def concat_mla_k(k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor) -> None:
    """
    Concatenate k_nope and k_rope into k for MLA (Multi-head Latent Attention).

    This kernel efficiently broadcasts k_rope across all heads while copying
    k_nope values directly.

    Args:
        k: Output tensor of shape [num_tokens, num_heads=128, k_head_dim=192], dtype=bfloat16
        k_nope: Input tensor of shape [num_tokens, num_heads=128, nope_head_dim=128], dtype=bfloat16
        k_rope: Input tensor of shape [num_tokens, 1, rope_head_dim=64], dtype=bfloat16
    """
    module = _jit_concat_mla_k_module()
    module.concat_mla_k(k, k_nope, k_rope)


def concat_mla_absorb_q(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> None:
    """
    Concatenate tensors a and b into out for MLA absorbed Q computation.

    Args:
        a: Input tensor of shape [dim_0, dim_1, 512], dtype=bfloat16
        b: Input tensor of shape [dim_0, dim_1, 64], dtype=bfloat16
        out: Output tensor of shape [dim_0, dim_1, 576], dtype=bfloat16
    """
    module = _jit_concat_mla_absorb_q_module()
    module.concat_mla_absorb_q(a, b, out)
