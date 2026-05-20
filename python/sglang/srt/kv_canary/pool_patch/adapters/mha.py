from __future__ import annotations

from typing import Literal

from sglang.jit_kernel.kv_canary_verify import RealKvSource
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.pool_patch.api import register_canary_adapter
from sglang.srt.kv_canary.pool_patch.utils import (
    _make_row_source,
    _patch_buf_info_method,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MHATokenToKVPoolFP4


@register_canary_adapter(MHATokenToKVPool)
class _MHAAdapter:
    def is_swa(self, pool: MHATokenToKVPool) -> bool:
        return False

    def has_v_half(self, pool: MHATokenToKVPool) -> bool:
        return True

    def build_real_kv_sources(
        self,
        pool: MHATokenToKVPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        buf = pool.k_buffer[0] if half == "K" else pool.v_buffer[0]
        return _make_row_source(layer_buffer=buf, read_bytes=read_bytes)

    def install_full_group(
        self, pool: MHATokenToKVPool, group: CanaryBufferGroup
    ) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_contiguous_buf_infos",
            group=group,
            has_v_half=True,
            page_size=pool.page_size,
        )

    def install_swa_group(
        self, pool: MHATokenToKVPool, group: CanaryBufferGroup
    ) -> None:
        raise NotImplementedError(
            f"kv-canary: MHA pool {type(pool).__name__} has no SWA sub-pool"
        )


@register_canary_adapter(MHATokenToKVPoolFP4)
class _MHAFp4Adapter(_MHAAdapter):
    pass
