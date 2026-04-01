from __future__ import annotations

from typing import Optional

from sglang.srt.mem_cache.codec.l2_codec_config import (
    HiCacheL2CodecConfig,
    parse_hicache_l2_codec_config,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool, NSATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSATokenToKVPoolHost,
)


def create_hicache_host_pool(
    *,
    device_pool,
    host_to_device_ratio: float,
    host_size: int,
    page_size: int,
    layout: str,
    allocator_type: str,
    storage_backend_extra_config: Optional[str],
):
    l2_cfg: Optional[HiCacheL2CodecConfig] = parse_hicache_l2_codec_config(
        storage_backend_extra_config
    )

    if isinstance(device_pool, MHATokenToKVPool):
        if l2_cfg and l2_cfg.name == "zlib":
            from sglang.srt.mem_cache.codec.mha_host_pool_zlib import (
                MHATokenToKVPoolHostZlib,
            )

            return MHATokenToKVPoolHostZlib(
                device_pool,
                host_to_device_ratio,
                host_size,
                page_size,
                layout,
                l2_codec=l2_cfg,
                allocator_type=allocator_type,
            )
        return MHATokenToKVPoolHost(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            allocator_type=allocator_type,
        )

    if isinstance(device_pool, NSATokenToKVPool):
        if l2_cfg:
            raise ValueError("L2 codec host pool is not implemented for NSA yet.")
        return NSATokenToKVPoolHost(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            allocator_type=allocator_type,
        )

    if isinstance(device_pool, MLATokenToKVPool):
        if l2_cfg:
            raise ValueError("L2 codec host pool is not implemented for MLA yet.")
        return MLATokenToKVPoolHost(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            allocator_type=allocator_type,
        )

    raise ValueError("Unsupported device_pool type for hierarchical cache host pool.")


