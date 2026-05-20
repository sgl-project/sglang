from __future__ import annotations

from typing import Literal

from sglang.jit_kernel.kv_canary_verify import RealKvSource
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.pool_patch.api import register_canary_adapter
from sglang.srt.kv_canary.pool_patch.utils import (
    _make_row_source,
    _patch_buf_info_method,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool


@register_canary_adapter(SWAKVPool)
class _SWAAdapter:
    def is_swa(self, pool: SWAKVPool) -> bool:
        return True

    def has_v_half(self, pool: SWAKVPool) -> bool:
        return True

    def build_real_kv_sources(
        self,
        pool: SWAKVPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        sub_pool = pool.full_kv_pool if kind is PoolKind.FULL else pool.swa_kv_pool
        buf = sub_pool.k_buffer[0] if half == "K" else sub_pool.v_buffer[0]
        return _make_row_source(layer_buffer=buf, read_bytes=read_bytes)

    def install_full_group(self, pool: SWAKVPool, group: CanaryBufferGroup) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_contiguous_buf_infos",
            group=group,
            has_v_half=True,
            page_size=pool.page_size,
        )

    def install_swa_group(self, pool: SWAKVPool, group: CanaryBufferGroup) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_state_buf_infos",
            group=group,
            has_v_half=True,
            page_size=pool.page_size,
        )
