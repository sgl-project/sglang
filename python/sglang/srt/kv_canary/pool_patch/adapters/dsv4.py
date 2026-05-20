from __future__ import annotations

from typing import Any, Callable, Literal

from sglang.jit_kernel.kv_canary_verify import RealKvSource
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.pool_patch.api import register_canary_adapter
from sglang.srt.kv_canary.pool_patch.helpers import (
    _BufInfoTriple,
    _make_packed_source,
    _make_row_source,
    _splice_packed_buf_info,
    _wrap_method,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


@register_canary_adapter(DeepSeekV4TokenToKVPool)
class _DeepSeekV4Adapter:
    def is_swa(self, pool: DeepSeekV4TokenToKVPool) -> bool:
        return True

    def has_v_half(self, pool: DeepSeekV4TokenToKVPool) -> bool:
        return False

    def build_real_kv_sources(
        self,
        pool: DeepSeekV4TokenToKVPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        if half == "V":
            return ()
        if kind is PoolKind.SWA:
            swa_buf = pool.swa_kv_pool.kv_buffer[0]
            return _make_row_source(layer_buffer=swa_buf, read_bytes=read_bytes)

        c4_buf = pool.c4_kv_pool.kv_buffer[0]
        c4_page_size = pool.c4_kv_pool.page_size
        c4_bytes_per_token = pool.c4_kv_pool.get_bytes_per_token()
        c4_sources = _make_packed_source(
            page_buffer=c4_buf,
            page_size=c4_page_size,
            bytes_per_token=c4_bytes_per_token,
            read_bytes=read_bytes,
        )

        indexer_buf = pool.c4_indexer_kv_pool.index_k_with_scale_buffer[0]
        indexer_page_size = pool.c4_indexer_kv_pool.page_size
        indexer_bytes_per_token = int(indexer_buf.shape[1]) // indexer_page_size
        indexer_sources = _make_packed_source(
            page_buffer=indexer_buf,
            page_size=indexer_page_size,
            bytes_per_token=indexer_bytes_per_token,
            read_bytes=read_bytes,
        )

        c128_buf = pool.c128_kv_pool.kv_buffer[0]
        c128_page_size = pool.c128_kv_pool.page_size
        c128_bytes_per_token = pool.c128_kv_pool.get_bytes_per_token()
        c128_sources = _make_packed_source(
            page_buffer=c128_buf,
            page_size=c128_page_size,
            bytes_per_token=c128_bytes_per_token,
            read_bytes=read_bytes,
        )

        return c4_sources + indexer_sources + c128_sources

    def install_full_group(
        self, pool: DeepSeekV4TokenToKVPool, group: CanaryBufferGroup
    ) -> None:
        _patch_dsv4_contiguous_buf_info(pool, group=group)

    def install_swa_group(
        self, pool: DeepSeekV4TokenToKVPool, group: CanaryBufferGroup
    ) -> None:
        _patch_dsv4_state_buf_info(pool, group=group)


def _patch_dsv4_contiguous_buf_info(
    pool: DeepSeekV4TokenToKVPool,
    *,
    group: CanaryBufferGroup,
) -> None:
    c4_layer_num = len(pool.c4_kv_pool.kv_buffer)
    indexer_layer_num = len(pool.c4_indexer_kv_pool.index_k_with_scale_buffer)
    c128_layer_num = len(pool.c128_kv_pool.kv_buffer)
    segment_offsets = [
        0,
        c4_layer_num,
        c4_layer_num + indexer_layer_num,
    ]
    expected_total = c4_layer_num + indexer_layer_num + c128_layer_num

    page_size = pool.page_size

    def _with_splice(original: Callable, *args: Any, **kwargs: Any) -> _BufInfoTriple:
        ptrs, lens, item_lens = original(*args, **kwargs)
        if len(ptrs) != expected_total:
            raise RuntimeError(
                f"DSV4 buf_info layout drifted: got {len(ptrs)}, expected {expected_total}"
            )
        return _splice_packed_buf_info(
            ptrs=ptrs,
            lens=lens,
            item_lens=item_lens,
            segment_offsets=segment_offsets,
            group=group,
            page_size=page_size,
        )

    _wrap_method(pool, "get_contiguous_buf_infos", wrapper=_with_splice)


def _patch_dsv4_state_buf_info(
    pool: DeepSeekV4TokenToKVPool,
    *,
    group: CanaryBufferGroup,
) -> None:
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
    segment_offsets = [
        0,
        swa_layer_num,
        swa_layer_num + compress_state_count,
    ]
    expected_total = swa_layer_num + compress_state_count + indexer_compress_state_count

    page_size = pool.page_size

    def _with_splice(original: Callable, *args: Any, **kwargs: Any) -> _BufInfoTriple:
        ptrs, lens, item_lens = original(*args, **kwargs)
        if len(ptrs) != expected_total:
            raise RuntimeError(
                f"DSV4 state buf_info layout drifted: got {len(ptrs)}, expected {expected_total}"
            )
        return _splice_packed_buf_info(
            ptrs=ptrs,
            lens=lens,
            item_lens=item_lens,
            segment_offsets=segment_offsets,
            group=group,
            page_size=page_size,
        )

    _wrap_method(pool, "get_state_buf_infos", wrapper=_with_splice)
