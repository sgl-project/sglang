from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_fused_qknorm_rope_module(head_dim: int, is_neox: bool, yarn: bool) -> Module:
    return load_jit(
        "fused_qknorm_rope",
        head_dim,
        int(is_neox),
        int(yarn),
        cuda_files=["elementwise/fused_qknorm_rope.cuh"],
        cuda_wrappers=[("fused_qk_norm_rope", "fused_qk_norm_rope")],
        extra_cuda_cflags=[
            "--use_fast_math",
            f"-DJIT_HEAD_DIM={head_dim}",
            f"-DJIT_INTERLEAVE={0 if is_neox else 1}",
            f"-DJIT_YARN={1 if yarn else 0}",
        ],
    )


@register_custom_op(
    op_name="fused_qk_norm_rope_out",
    mutates_args=["qkv"],
)
def fused_qk_norm_rope_out(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    position_ids: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    base: float,
    is_neox: bool,
    factor: float,
    low: float,
    high: float,
    attention_factor: float,
    rotary_dim: int,
) -> None:
    """
    Fused QK RMSNorm + RoPE applied in-place on the QKV tensor.

    Matches the call signature of ``sgl_kernel.fused_qk_norm_rope``.

    Args:
        qkv:              [num_tokens, (nq+nk+nv)*head_dim] bfloat16 — modified in-place
        q_weight:         [head_dim] bfloat16 — RMSNorm weights for Q
        k_weight:         [head_dim] bfloat16 — RMSNorm weights for K
        position_ids:     [num_tokens] int32
        num_heads_q:      number of query heads
        num_heads_k:      number of key heads
        num_heads_v:      number of value heads
        head_dim:         head dimension; must be 64, 128, or 256
        eps:              epsilon for RMSNorm
        base:             RoPE base frequency
        is_neox:          True → NeoX style, False → interleave (GPT-J) style
        factor:           YaRN scaling factor (1.0 = standard RoPE)
        low:              YaRN low-frequency threshold
        high:             YaRN high-frequency threshold
        attention_factor: scale applied to the rotary component
        rotary_dim:       number of elements per head to apply RoPE to
    """
    yarn = factor != 1.0
    module = _jit_fused_qknorm_rope_module(head_dim, is_neox, yarn)
    module.fused_qk_norm_rope(
        qkv,
        q_weight,
        k_weight,
        position_ids,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        base,
        1 if is_neox else 0,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )


@cache_once
def can_use_fused_qk_norm_rope(
    head_dim: int, is_neox: bool, dtype: torch.dtype, yarn: bool = False
) -> bool:
    """Return True if the JIT fused QK-Norm + RoPE kernel can be used.

    Args:
        head_dim: head dimension; supported values are 64, 128, 256
        dtype: tensor dtype; only bfloat16 is supported
        yarn: whether YaRN scaling is active (factor != 1.0); prebuilds the
              correct kernel variant so no extra JIT compile occurs on the
              first real call.
    """
    logger = logging.getLogger(__name__)
    if head_dim not in (64, 128, 256):
        logger.warning(
            f"Unsupported head_dim={head_dim} for JIT fused_qk_norm_rope kernel"
        )
        return False
    if dtype != torch.bfloat16:
        logger.warning(f"Unsupported dtype={dtype} for JIT fused_qk_norm_rope kernel")
        return False
    try:
        _jit_fused_qknorm_rope_module(head_dim, is_neox, yarn)
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT fused_qk_norm_rope kernel: {e}")
        return False


def fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor,
    factor: float,
    low: float,
    high: float,
    attention_factor: float,
    rotary_dim: Optional[int] = None,
) -> None:
    """
    Fused QK RMSNorm + RoPE applied in-place on the QKV tensor.

    Matches the call signature of ``sgl_kernel.fused_qk_norm_rope``.

    Args:
        qkv:              [num_tokens, (nq+nk+nv)*head_dim] bfloat16 — modified in-place
        num_heads_q:      number of query heads
        num_heads_k:      number of key heads
        num_heads_v:      number of value heads
        head_dim:         head dimension; must be 64, 128, or 256
        eps:              epsilon for RMSNorm
        q_weight:         [head_dim] bfloat16 — RMSNorm weights for Q
        k_weight:         [head_dim] bfloat16 — RMSNorm weights for K
        base:             RoPE base frequency
        is_neox:          True → NeoX style, False → interleave (GPT-J) style
        position_ids:     [num_tokens] int32
        factor:           YaRN scaling factor (1.0 = standard RoPE)
        low:              YaRN low-frequency threshold
        high:             YaRN high-frequency threshold
        attention_factor: scale applied to the rotary component
        rotary_dim:       elements per head to rotate; defaults to head_dim
    """
    if rotary_dim is None:
        rotary_dim = head_dim
    fused_qk_norm_rope_out(
        qkv,
        q_weight,
        k_weight,
        position_ids,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        base,
        is_neox,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )
