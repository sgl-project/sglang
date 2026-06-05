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
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)


@cache_once
def _jit_qknorm_rope_module(
    head_dim: int,
    rope_dim: int,
    is_neox: bool,
    dtype: torch.dtype,
) -> Module:
    args = make_cpp_args(head_dim, rope_dim, is_neox, is_arch_support_pdl(), dtype)
    return load_jit(
        "qknorm_rope",
        *args,
        cuda_files=["diffusion/qknorm_rope.cuh"],
        cuda_wrappers=[("qknorm_rope", f"QKNormRopeKernel<{args}>::run")],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_inplace_qknorm_rope(
    head_dim: int,
    rope_dim: int,
    is_neox: bool,
    dtype: torch.dtype,
) -> bool:
    if head_dim not in (64, 128, 256):
        logger.warning(f"Unsupported head_dim={head_dim} for JIT fused QKNorm+RoPE")
        return False
    if rope_dim <= 0 or rope_dim > head_dim:
        logger.warning(
            f"Unsupported rope_dim={rope_dim} for head_dim={head_dim} in fused QKNorm+RoPE"
        )
        return False
    elems_per_thread = head_dim // 32
    if rope_dim % elems_per_thread != 0:
        logger.warning(
            "rope_dim=%s must be divisible by per-thread width=%s for fused QKNorm+RoPE",
            rope_dim,
            elems_per_thread,
        )
        return False
    if is_neox:
        rotary_lanes = rope_dim // elems_per_thread
        if rotary_lanes < 2 or rotary_lanes & (rotary_lanes - 1):
            logger.warning(
                "rope_dim=%s yields invalid rotary_lanes=%s for neox fused QKNorm+RoPE; rotary lane count must be a power of 2",
                rope_dim,
                rotary_lanes,
            )
            return False
    try:
        _jit_qknorm_rope_module(head_dim, rope_dim, is_neox, dtype)
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT fused QKNorm+RoPE kernel: {e}")
        return False


@register_custom_op(mutates_args=["q", "k"])
def fused_inplace_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    eps: float = 1e-6,
    head_dim: int = 0,
    rope_dim: int = 0,
) -> None:
    head_dim = head_dim or q.size(-1)
    rope_dim = rope_dim or cos_sin_cache.size(-1)
    module = _jit_qknorm_rope_module(head_dim, rope_dim, is_neox, q.dtype)
    module.qknorm_rope(q, k, q_weight, k_weight, cos_sin_cache, positions, eps)
