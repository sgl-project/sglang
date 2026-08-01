# Copyright (c) 2026, SGLang Team.
"""Pure workload qualification shared by the SM120 FA4 host and kernel."""

from typing import Optional

LOW_HD_DECODE_SHAPES = frozenset({(64, 64), (128, 128)})
LOW_HD_DECODE_TILE_N = 64
LOW_HD_DECODE_MIN_VISIBLE_K = 256
LOW_HD_DECODE_SHORT_VISIBLE_K = 512
LOW_HD_DECODE_MAX_SPLITS = 8
LOW_HD_DECODE_SHORT_MIN_TILES_PER_CTA = 2
LOW_HD_DECODE_LONG_MIN_TILES_PER_CTA = 8


def visible_decode_seqlen_k(
    max_seqlen_k: int,
    *,
    is_local: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
) -> int:
    """Return the exact KV span visible to a single query position."""
    if not is_local:
        return max(0, max_seqlen_k)
    left = max_seqlen_k if window_size_left is None else max(0, window_size_left)
    right = max_seqlen_k if window_size_right is None else max(0, window_size_right)
    return max(0, min(max_seqlen_k, left + 1 + right))


def low_hd_paged_decode_tile_m(
    *,
    head_dim: int,
    head_dim_v: int,
    paged_kv: bool,
    seqlen_q: Optional[int],
    visible_seqlen_k: Optional[int],
    qhead_per_kvhead: Optional[int],
    num_sms: Optional[int] = None,
    total_mblocks: Optional[int] = None,
) -> Optional[int]:
    """Return the qualified low-HD decode M tile, or ``None`` for fallback."""
    if (
        not paged_kv
        or seqlen_q != 1
        or visible_seqlen_k is None
        or visible_seqlen_k <= LOW_HD_DECODE_MIN_VISIBLE_K
        or (head_dim, head_dim_v) not in LOW_HD_DECODE_SHAPES
    ):
        return None
    qhead_ratio = 1 if qhead_per_kvhead is None else qhead_per_kvhead
    if (
        (head_dim, head_dim_v) == (64, 64)
        and qhead_ratio >= 8
        and visible_seqlen_k <= LOW_HD_DECODE_SHORT_VISIBLE_K
    ):
        if num_sms is not None and total_mblocks is not None:
            num_n_blocks = (
                visible_seqlen_k + LOW_HD_DECODE_TILE_N - 1
            ) // LOW_HD_DECODE_TILE_N
            max_short_splits = min(
                LOW_HD_DECODE_MAX_SPLITS,
                num_n_blocks // LOW_HD_DECODE_SHORT_MIN_TILES_PER_CTA,
            )
            # Use the one-warp M16 CTA only when bounded SplitKV can fill an
            # SM wave without reducing each partition below two KV tiles.
            if total_mblocks * max_short_splits >= num_sms:
                return 16
        return None
    if qhead_ratio >= 8 and visible_seqlen_k <= LOW_HD_DECODE_SHORT_VISIBLE_K:
        return 32
    return 16


def is_low_hd_paged_decode_tile(
    *,
    head_dim: int,
    head_dim_v: int,
    paged_kv: bool,
    seqlen_q: int,
    tile_m: int,
    tile_n: int,
) -> bool:
    """Return whether a selected tile belongs to the qualified low-HD path."""
    return (
        paged_kv
        and seqlen_q == 1
        and (head_dim, head_dim_v) in LOW_HD_DECODE_SHAPES
        and tile_m in (16, 32)
        and tile_n == LOW_HD_DECODE_TILE_N
    )
