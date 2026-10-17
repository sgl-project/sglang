"""Fused Qwen3-Next Q/K Gemma RMSNorm and partial RoPE.

The token count selects the kernel. KV-cache writes stay off; attention
stores K and V itself.
"""

from __future__ import annotations

import torch

_M64_MAX = 8192
_M128_MAX = 24576
_PREFILL_MAX = 32768
_PACKED_WIDTH = 2560
_ROTARY_DIM = 64

_gluon_available: bool | None = None


def gluon_available() -> bool:
    """True when Triton's Gluon frontend can be imported."""
    global _gluon_available
    if _gluon_available is None:
        try:
            from triton.experimental import gluon  # noqa: F401

            _gluon_available = True
        except Exception:
            _gluon_available = False
    return _gluon_available


def prologue_kind(num_tokens: int) -> str | None:
    """Kernel name for this token count, or None when the stock prepare should run."""
    if num_tokens <= _M64_MAX:
        return "m64"
    if not gluon_available():
        return None
    if num_tokens <= _M128_MAX:
        return "m128"
    if num_tokens <= _PREFILL_MAX:
        return "prefill"
    return None


def fused_qwen3_next_qk_norm_rope(
    projected_qkv_gate: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    rotary_dim: int = _ROTARY_DIM,
    key_cache: torch.Tensor | None = None,
    value_cache: torch.Tensor | None = None,
    cache_locations: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return Q, K, V, and gate, each shaped ``(tokens, hidden)``.

    ``None`` means this shape is not covered. Cache arguments are accepted so
    callers can prove they are left unchanged; the kernels are launched with
    ``store_kv=False`` and never index them.
    """
    rows, width = projected_qkv_gate.shape
    if width != _PACKED_WIDTH or rotary_dim != _ROTARY_DIM:
        return None
    kind = prologue_kind(rows)
    if kind is None:
        return None

    if key_cache is None:
        key_cache = projected_qkv_gate
    if value_cache is None:
        value_cache = projected_qkv_gate
    if cache_locations is None:
        cache_locations = positions

    if kind == "m64":
        from .attention_prologue_m64 import (
            fused_attention_qk_norm_rope_kv_cache as implementation,
        )
    elif kind == "m128":
        from .attention_prologue_m128_m256 import (
            fused_attention_qk_norm_rope_kv_cache as implementation,
        )
    else:
        from .attention_prologue_prefill_1024_32768 import (
            fused_attention_qk_norm_rope_kv_cache as implementation,
        )

    query, gate, key, value, _, _ = implementation(
        projected_qkv_gate,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        cache_locations,
        key_cache,
        value_cache,
        eps=eps,
        rotary_dim=rotary_dim,
        store_kv=False,
    )
    return (
        query.reshape(rows, -1),
        key.reshape(rows, -1),
        value.reshape(rows, -1),
        gate.reshape(rows, -1),
    )
