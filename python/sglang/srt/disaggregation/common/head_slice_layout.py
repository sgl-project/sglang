"""Byte ranges that move one destination rank's KV heads between attention TP sizes."""

from __future__ import annotations

from typing import List, Tuple

import msgspec
import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.common.staging_buffer import compute_head_slice_params


class HeadSliceSegment(msgspec.Struct, frozen=True):
    src_offset: int
    dst_offset: int
    num_bytes: int


def _validate_head_split(
    *, total_kv_heads: int, src_attn_tp_size: int, dst_attn_tp_size: int
) -> None:
    if total_kv_heads % src_attn_tp_size or total_kv_heads % dst_attn_tp_size:
        raise ValueError(
            "head-sliced PD transfer needs KV heads divisible by both attention TP "
            f"sizes, got heads={total_kv_heads}, prefill TP={src_attn_tp_size}, "
            f"decode TP={dst_attn_tp_size}"
        )


def page_row_head_segments(
    *,
    total_kv_heads: int,
    head_dim: int,
    page_size: int,
    src_attn_tp_size: int,
    dst_attn_tp_size: int,
    src_tp_rank: int,
    dst_tp_rank: int,
) -> Tuple[List[HeadSliceSegment], int, int]:
    """Segments inside one page row of the MSA-v1 NVFP4 main KV pool.

    Returns the segments, then the source and destination row widths in bytes.
    """
    from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import (
        main_page_components,
        page_row_layout,
    )

    _validate_head_split(
        total_kv_heads=total_kv_heads,
        src_attn_tp_size=src_attn_tp_size,
        dst_attn_tp_size=dst_attn_tp_size,
    )
    src_head_start, num_heads, dst_head_start, _ = compute_head_slice_params(
        src_attn_tp_size, dst_attn_tp_size, src_tp_rank, dst_tp_rank, total_kv_heads
    )
    src_components = main_page_components(
        total_kv_heads // src_attn_tp_size, head_dim, page_size
    )
    dst_components = main_page_components(
        total_kv_heads // dst_attn_tp_size, head_dim, page_size
    )
    src_offsets, src_row_bytes = page_row_layout(src_components)
    dst_offsets, dst_row_bytes = page_row_layout(dst_components)
    segments = []
    for (_, (_, tokens, bytes_per_token)), src_base, dst_base in zip(
        src_components, src_offsets, dst_offsets
    ):
        # Each component is [heads, page_size, bytes], so one head is a contiguous run.
        head_bytes = tokens * bytes_per_token
        segments.append(
            HeadSliceSegment(
                src_offset=src_base + src_head_start * head_bytes,
                dst_offset=dst_base + dst_head_start * head_bytes,
                num_bytes=num_heads * head_bytes,
            )
        )
    return segments, src_row_bytes, dst_row_bytes


def token_major_head_segments(
    *,
    total_kv_heads: int,
    src_item_len: int,
    dst_item_len: int,
    page_size: int,
    src_attn_tp_size: int,
    dst_attn_tp_size: int,
    src_tp_rank: int,
    dst_tp_rank: int,
) -> List[HeadSliceSegment]:
    """Segments inside one page of a [token, head, head_bytes] KV buffer."""
    _validate_head_split(
        total_kv_heads=total_kv_heads,
        src_attn_tp_size=src_attn_tp_size,
        dst_attn_tp_size=dst_attn_tp_size,
    )
    src_heads = total_kv_heads // src_attn_tp_size
    dst_heads = total_kv_heads // dst_attn_tp_size
    src_token_bytes = src_item_len // page_size
    dst_token_bytes = dst_item_len // page_size
    head_bytes = src_token_bytes // src_heads
    if (
        src_token_bytes * page_size != src_item_len
        or head_bytes * src_heads != src_token_bytes
        or head_bytes * dst_heads != dst_token_bytes
        or dst_token_bytes * page_size != dst_item_len
    ):
        raise ValueError(
            "token-major KV entry does not split into whole heads: "
            f"src item={src_item_len}, dst item={dst_item_len}, page={page_size}, "
            f"src heads={src_heads}, dst heads={dst_heads}"
        )
    src_head_start, num_heads, dst_head_start, _ = compute_head_slice_params(
        src_attn_tp_size, dst_attn_tp_size, src_tp_rank, dst_tp_rank, total_kv_heads
    )
    return [
        HeadSliceSegment(
            src_offset=token * src_token_bytes + src_head_start * head_bytes,
            dst_offset=token * dst_token_bytes + dst_head_start * head_bytes,
            num_bytes=num_heads * head_bytes,
        )
        for token in range(page_size)
    ]


def paged_segment_addresses(
    *,
    src_base_ptr: int,
    dst_base_ptr: int,
    src_page_indices: npt.NDArray,
    dst_page_indices: npt.NDArray,
    src_page_bytes: int,
    dst_page_bytes: int,
    segments: List[HeadSliceSegment],
) -> Tuple[List[int], List[int], List[int]]:
    """Source addresses, destination addresses and lengths, page by page."""
    src_offsets = np.array([s.src_offset for s in segments], dtype=np.int64)
    dst_offsets = np.array([s.dst_offset for s in segments], dtype=np.int64)
    lengths = np.array([s.num_bytes for s in segments], dtype=np.int64)
    src_pages = src_page_indices.astype(np.int64).reshape(-1, 1)
    dst_pages = dst_page_indices.astype(np.int64).reshape(-1, 1)
    src = src_base_ptr + src_pages * src_page_bytes + src_offsets.reshape(1, -1)
    dst = dst_base_ptr + dst_pages * dst_page_bytes + dst_offsets.reshape(1, -1)
    length = np.broadcast_to(lengths.reshape(1, -1), src.shape)
    return (
        src.reshape(-1).tolist(),
        dst.reshape(-1).tolist(),
        length.reshape(-1).tolist(),
    )


def draft_window_chunk_start(
    *, window_pages: int, chunk_page_start: int, total_pages: int
) -> int:
    """Index of the first page in a chunk whose draft KV decode can attend to.

    `window_pages` trailing pages of the request are sent; 0 sends every page.
    """
    if window_pages == 0:
        return 0
    return max(0, total_pages - window_pages - chunk_page_start)
