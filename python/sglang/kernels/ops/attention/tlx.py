"""gfx950 attention input preparation from fbtriton ``tlx.ops.qk_norm_rope_kv_cache``, vendored in ``sglang.kernels.tlx_ops``.

``qk_norm_rope_kv_cache`` computes what ``qk_norm_rope_cache`` (``kernels/ops/attention/qk_norm_rope_cache.py``)
does, in one launch for any number of tokens, with the per-head sums of squares in another fixed order: weightless
per-head RMSNorm of q and k with the BF16 round trips of the unfused sequence, the query scale, interleaved RoPE on
BF16-rounded cos/sin and products, and the FP8 K/V store into ``[num_blocks, page, HK, D]`` caches through the slot
mapping (optionally remapped full -> SWA; negative slots write nothing). It returns q only; k is consumed by the
cache store. Rejected (see ``qk_norm_rope_kv_cache_supported``): other dtypes, head dims or head counts that are not
powers of two, rows wider than 4096. The caller gates the architecture. The kernel is plain Triton (no TLX primitives), so stock ROCm Triton runs it.
"""

import torch


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def qk_norm_rope_kv_cache_supported(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor
) -> bool:
    """Whether ``qk_norm_rope_kv_cache`` handles these operands; host-only, never reads device data."""
    if k_cache.dim() != 4 or q.dim() != 2 or k.dim() != 2 or v.shape != k.shape:
        return False
    _, _, num_kv_heads, head_dim = k_cache.shape
    return (
        q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.stride(1) == k.stride(1) == v.stride(1) == 1
        and _pow2(head_dim)
        and 16 <= head_dim <= 256
        and q.shape[1] % head_dim == 0
        and _pow2(q.shape[1] // head_dim)
        and _pow2(num_kv_heads)
        and q.shape[1] <= 4096
        and k.shape == (q.shape[0], num_kv_heads * head_dim)
    )


def qk_norm_rope_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    slot_remap: torch.Tensor | None,
    positions: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    eps: float,
    q_scale: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> torch.Tensor:
    """Normalized, scaled and rotated BF16 q ``(T, HQ * D)``; K/V written to the FP8 caches."""
    from sglang.kernels.tlx_ops.kernels.fused.qk_norm_rope_kv_cache.gfx950 import (
        qk_norm_rope_kv_cache as _qk_norm_rope_kv_cache,
    )

    return _qk_norm_rope_kv_cache(
        q,
        k,
        v,
        k_cache,
        v_cache,
        slot_mapping,
        eps,
        q_scale,
        positions,
        cos_sin_cache,
        True,
        slot_remap,
        k_scale,
        v_scale,
        None,
        None,
    )
