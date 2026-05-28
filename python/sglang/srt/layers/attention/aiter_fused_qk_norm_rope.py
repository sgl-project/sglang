# SPDX-License-Identifier: Apache-2.0
"""SGLang-side wrapper around aiter's ``fused_qkv_split_qk_norm_rope_cache`` kernel.

This kernel performs, in a single launch:
    1. Split a packed ``qkv`` tensor into Q [+ gate], K, V slices.
    2. RMSNorm on per-head Q and K.
    3. Apply RoPE to Q and K (NeoX or GPT-J style; partial rotary supported).
    4. Write K and V to the paged KV cache (optional per-call FP8 scale).

It replaces SGLang's current sequence of:
    qkv_proj -> split -> _apply_qk_norm -> rotary_emb -> set_kv_buffer

When this path is taken, the caller MUST set ``save_kv_cache=False`` on the
subsequent ``self.attn(...)`` call because the fused kernel has already
populated the cache.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()

# Opt-in flag (default off until layout / scale wiring is validated end-to-end).
SGLANG_USE_AITER_FUSED_QK_NORM_ROPE = get_bool_env_var(
    "SGLANG_USE_AITER_FUSED_QK_NORM_ROPE_CACHE", "false"
)


def _try_import() -> Optional[callable]:
    """Lazy import so non-ROCm builds stay clean."""
    if not _is_hip:
        return None
    try:
        from aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache import (
            fused_qkv_split_qk_norm_rope_cache,
        )

        return fused_qkv_split_qk_norm_rope_cache
    except ImportError:
        return None


_FUSED_KERNEL = _try_import()


def is_available() -> bool:
    """Whether the fused kernel can be used at runtime."""
    return SGLANG_USE_AITER_FUSED_QK_NORM_ROPE and _FUSED_KERNEL is not None


def fused_qk_norm_rope_cache(
    qkv: torch.Tensor,
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    eps: float,
    is_neox: bool = True,
    attn_output_gate: bool = False,
    gated_qkv_layout: str = "interleaved",
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Split + QK-RMSNorm + RoPE + KV-cache-write in one kernel.

    Args:
        qkv: ``[T, packed_dim]`` packed Q[+gate]/K/V from a fused QKV projection.
        q_weight / k_weight: RMSNorm gammas ``(head_dim,)``.
        cos_sin_cache: SGLang ``RotaryEmbedding.cos_sin_cache`` ``[max_pos, head_dim]``.
            We split it in half on the last dim to obtain ``(cos, sin)``.
        positions: ``[T]`` int token positions for RoPE lookup.
        k_buffer / v_buffer: Flat per-token KV pool tensors
            ``[size+page_size, head_num, head_dim]`` returned by
            ``MHATokenToKVPool.get_key_buffer()`` / ``get_value_buffer()``.
            We reshape them to NHD paged layout
            ``[num_blocks, page_size, head_num, head_dim]``.
        slot_mapping: ``[T]`` token-absolute slot indices (``forward_batch.out_cache_loc``).
        num_q_heads / num_kv_heads / head_dim: Local (TP-sharded) head counts.
        page_size: Pool page size; with ``page_size=1`` this becomes a per-token
            slot layout.
        eps: RMSNorm epsilon.
        is_neox: NeoX-style rotation (default for Qwen3.5).
        attn_output_gate: If True, ``qkv`` carries ``Q || gate || K || V`` per the
            ``gated_qkv_layout`` (interleaved = Q/gate per head; blocked = all Q then all gate).
        k_scale / v_scale: Optional FP8 quant scalars applied before cache write.

    Returns:
        ``(q, gate, k, v)`` post-norm/post-rope tensors of shape
        ``[T, num_q_heads, head_dim]`` / ``[T, num_kv_heads, head_dim]``.
        ``gate`` is ``None`` when ``attn_output_gate=False``.

    The caller MUST pass ``save_kv_cache=False`` to ``RadixAttention.forward`` since
    K/V are already written to the cache.
    """
    if _FUSED_KERNEL is None:
        raise RuntimeError(
            "fused_qkv_split_qk_norm_rope_cache is not available "
            "(aiter not installed or non-ROCm build)."
        )

    # Reshape flat per-token pool to NHD paged layout expected by the kernel.
    # k_buffer / v_buffer shape: [size + page_size, head_num, head_dim].
    total_slots = k_buffer.shape[0]
    if total_slots % page_size != 0:
        raise ValueError(
            f"k_buffer first dim ({total_slots}) must be divisible by "
            f"page_size ({page_size}); check pool allocation."
        )
    num_blocks = total_slots // page_size
    key_cache = k_buffer.view(num_blocks, page_size, k_buffer.shape[1], k_buffer.shape[2])
    value_cache = v_buffer.view(
        num_blocks, page_size, v_buffer.shape[1], v_buffer.shape[2]
    )

    # Split combined cos_sin_cache into cos, sin. SGLang stores them concatenated
    # along the last dim as [max_pos, head_dim] = [max_pos, rotary_dim/2 + rotary_dim/2].
    cos, sin = cos_sin_cache.chunk(2, dim=-1)

    result = _FUSED_KERNEL(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        cos=cos,
        sin=sin,
        positions=positions,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        qh=num_q_heads,
        kvh=num_kv_heads,
        head_dim=head_dim,
        is_neox=is_neox,
        attn_output_gate=attn_output_gate,
        gated_qkv_layout=gated_qkv_layout,
        kv_cache_layout="NHD",
        k_scale=k_scale,
        v_scale=v_scale,
        eps=eps,
    )

    if attn_output_gate:
        q, gate, k, v = result
        return q, gate, k, v
    q, k, v = result
    return q, None, k, v
