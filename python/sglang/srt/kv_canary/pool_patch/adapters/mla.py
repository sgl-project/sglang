from __future__ import annotations

from typing import Literal

from sglang.jit_kernel.kv_canary_verify import RealKvSource
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.pool_patch.api import register_canary_adapter
from sglang.srt.kv_canary.pool_patch.utils import (
    _make_packed_source,
    _make_row_source,
    _patch_buf_info_method,
)
from sglang.srt.mem_cache.memory_pool import (
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NSATokenToKVPool,
)


@register_canary_adapter(MLATokenToKVPool)
class _MLAAdapter:
    def is_swa(self, pool: MLATokenToKVPool) -> bool:
        return False

    def has_v_half(self, pool: MLATokenToKVPool) -> bool:
        return False

    def build_real_kv_sources(
        self,
        pool: MLATokenToKVPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        if half == "V":
            return ()
        return _make_row_source(layer_buffer=pool.kv_buffer[0], read_bytes=read_bytes)

    def install_full_group(
        self, pool: MLATokenToKVPool, group: CanaryBufferGroup
    ) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_contiguous_buf_infos",
            group=group,
            has_v_half=False,
            page_size=pool.page_size,
        )

    def install_swa_group(
        self, pool: MLATokenToKVPool, group: CanaryBufferGroup
    ) -> None:
        raise NotImplementedError(
            f"kv-canary: MLA pool {type(pool).__name__} has no SWA sub-pool"
        )


@register_canary_adapter(MLATokenToKVPoolFP4)
class _MLAFp4Adapter(_MLAAdapter):
    pass


@register_canary_adapter(NSATokenToKVPool)
class _NSAAdapter(_MLAAdapter):
    def build_real_kv_sources(
        self,
        pool: NSATokenToKVPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        if half == "V":
            return ()
        kv_sources = _make_row_source(
            layer_buffer=pool.kv_buffer[0], read_bytes=read_bytes
        )
        index_buffer = pool.index_k_with_scale_buffer[0]
        index_page_size = pool.page_size
        index_bytes_per_token = int(index_buffer.shape[1]) // index_page_size
        index_sources = _make_packed_source(
            page_buffer=index_buffer,
            page_size=index_page_size,
            bytes_per_token=index_bytes_per_token,
            read_bytes=read_bytes,
        )
        return kv_sources + index_sources
