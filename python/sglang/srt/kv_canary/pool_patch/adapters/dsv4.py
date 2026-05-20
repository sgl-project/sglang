from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.pool_patch.utils import (
    BufInfoTriple,
    alloc_canary_buf,
    ensure_swa_lut_int32,
    make_packed_source,
    make_row_source,
    splice_segmented_buf_info,
    swa_index_lut,
    wrap_method,
)


def attach_dsv4(
    *,
    pool: object,
    device: torch.device,
    read_bytes: int,
    allocator: Optional[object] = None,
) -> tuple[CanaryBufferGroup, ...]:
    """Attach canary buffers to a DSV4 packed pool.

    FULL group covers three segments (c4 / indexer / c128) and splices into ``get_contiguous_buf_infos``;
    SWA group is a single-segment row pool and splices into ``get_state_buf_infos``.
    """
    ensure_swa_lut_int32(pool=pool, allocator=allocator)

    full_group = _build_full_group(pool=pool, device=device, read_bytes=read_bytes)
    swa_group = _build_swa_group(pool=pool, device=device, read_bytes=read_bytes)

    _patch_contiguous_buf_info(pool, group=full_group)
    _patch_state_buf_info(pool, group=swa_group)

    return (full_group, swa_group)


def _build_full_group(
    *,
    pool: object,
    device: torch.device,
    read_bytes: int,
) -> CanaryBufferGroup:
    c4_pool = pool.c4_kv_pool
    indexer_pool = pool.c4_indexer_kv_pool
    c128_pool = pool.c128_kv_pool

    num_slots = int(c4_pool.kv_buffer[0].shape[0]) * c4_pool.page_size
    k_head = alloc_canary_buf(num_slots=num_slots, device=device)
    k_tail = alloc_canary_buf(num_slots=num_slots, device=device)

    indexer_buf = indexer_pool.index_k_with_scale_buffer[0]
    indexer_bytes_per_token = int(indexer_buf.shape[1]) // indexer_pool.page_size

    sources = (
        make_packed_source(
            page_buffer=c4_pool.kv_buffer[0],
            page_size=c4_pool.page_size,
            bytes_per_token=c4_pool.get_bytes_per_token(),
            read_bytes=read_bytes,
        )
        + make_packed_source(
            page_buffer=indexer_buf,
            page_size=indexer_pool.page_size,
            bytes_per_token=indexer_bytes_per_token,
            read_bytes=read_bytes,
        )
        + make_packed_source(
            page_buffer=c128_pool.kv_buffer[0],
            page_size=c128_pool.page_size,
            bytes_per_token=c128_pool.get_bytes_per_token(),
            read_bytes=read_bytes,
        )
    )

    return CanaryBufferGroup(
        kind=PoolKind.FULL,
        k_head=k_head,
        k_tail=k_tail,
        v_head=None,
        v_tail=None,
        real_kv_sources_k=sources,
        real_kv_sources_v=(),
        swa_index_lut=None,
    )


def _build_swa_group(
    *,
    pool: object,
    device: torch.device,
    read_bytes: int,
) -> CanaryBufferGroup:
    swa_pool = pool.swa_kv_pool
    num_slots = int(swa_pool.kv_buffer[0].shape[0]) * swa_pool.page_size
    k_head = alloc_canary_buf(num_slots=num_slots, device=device)
    k_tail = alloc_canary_buf(num_slots=num_slots, device=device)
    return CanaryBufferGroup(
        kind=PoolKind.SWA,
        k_head=k_head,
        k_tail=k_tail,
        v_head=None,
        v_tail=None,
        real_kv_sources_k=make_row_source(
            layer_buffer=swa_pool.kv_buffer[0], read_bytes=read_bytes
        ),
        real_kv_sources_v=(),
        swa_index_lut=swa_index_lut(pool),
    )


def _patch_contiguous_buf_info(pool: object, *, group: CanaryBufferGroup) -> None:
    c4_layer_num = len(pool.c4_kv_pool.kv_buffer)
    indexer_layer_num = len(pool.c4_indexer_kv_pool.index_k_with_scale_buffer)
    c128_layer_num = len(pool.c128_kv_pool.kv_buffer)
    segment_offsets: List[int] = [
        0,
        c4_layer_num,
        c4_layer_num + indexer_layer_num,
    ]
    expected_total = c4_layer_num + indexer_layer_num + c128_layer_num
    page_size = pool.page_size

    def _with_splice(original: Callable, *args: Any, **kwargs: Any) -> BufInfoTriple:
        ptrs, lens, item_lens = original(*args, **kwargs)
        if len(ptrs) != expected_total:
            raise RuntimeError(
                f"DSV4 buf_info layout drifted: got {len(ptrs)}, expected {expected_total}"
            )
        return splice_segmented_buf_info(
            ptrs=ptrs,
            lens=lens,
            item_lens=item_lens,
            segment_offsets=segment_offsets,
            group=group,
            page_size=page_size,
        )

    wrap_method(pool, "get_contiguous_buf_infos", wrapper=_with_splice)


def _patch_state_buf_info(pool: object, *, group: CanaryBufferGroup) -> None:
    swa_layer_num = len(pool.swa_kv_pool.kv_buffer)
    compress_state_count = sum(1 for p in pool.compress_state_pools if p is not None)
    indexer_compress_state_count = sum(
        1 for p in pool.indexer_compress_state_pools if p is not None
    )
    if compress_state_count == 0 and indexer_compress_state_count == 0:
        raise NotImplementedError(
            "kv-canary: DSV4 SWA segmentation has empty compress_state_pools and "
            "indexer_compress_state_pools — cannot splice head/tail canary per segment"
        )
    segment_offsets: List[int] = [
        0,
        swa_layer_num,
        swa_layer_num + compress_state_count,
    ]
    expected_total = swa_layer_num + compress_state_count + indexer_compress_state_count
    page_size = pool.page_size

    def _with_splice(original: Callable, *args: Any, **kwargs: Any) -> BufInfoTriple:
        ptrs, lens, item_lens = original(*args, **kwargs)
        if len(ptrs) != expected_total:
            raise RuntimeError(
                f"DSV4 state buf_info layout drifted: got {len(ptrs)}, expected {expected_total}"
            )
        return splice_segmented_buf_info(
            ptrs=ptrs,
            lens=lens,
            item_lens=item_lens,
            segment_offsets=segment_offsets,
            group=group,
            page_size=page_size,
        )

    wrap_method(pool, "get_state_buf_infos", wrapper=_with_splice)
