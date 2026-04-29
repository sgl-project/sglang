from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostPoolGroup,
    MambaPoolHost,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSAIndexerPoolHost,
    PoolEntry,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def build_nsa_hybrid_stack(
    radix_cache: "HiRadixCache",
    params: "CacheInitParams",
    server_args: "ServerArgs",
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
) -> None:
    """HostPoolGroup (KV + indexer) + HybridCacheController for NSA (DSA)."""
    try:
        kv = radix_cache.kv_cache
        mla_host = MLATokenToKVPoolHost(
            kv,
            server_args.hicache_ratio,
            server_args.hicache_size,
            radix_cache.page_size,
            server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
            override_kv_cache_dim=kv.kv_cache_dim,
        )
        indexer_host = NSAIndexerPoolHost(
            kv,
            mla_host,
            server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        layer_num = kv.layer_num

        def layer_mapper(layer_id: int):
            if 0 <= layer_id < layer_num:
                return layer_id
            return None

        host_pool_group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=mla_host,
                    device_pool=kv,
                    layer_mapper=layer_mapper,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name=PoolName.INDEXER,
                    host_pool=indexer_host,
                    device_pool=kv,
                    layer_mapper=layer_mapper,
                    share_indices_with_anchor=True,
                ),
            ]
        )
        cache_controller = HybridCacheController(
            params.token_to_kv_pool_allocator,
            host_pool_group,
            radix_cache.page_size,
            radix_cache.tp_group,
            load_cache_event=load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=radix_cache.pp_rank,
            pp_size=radix_cache.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            transfer_layer_num=layer_num,
            enable_storage_metrics=enable_storage_metrics,
        )
        radix_cache.full_kv_pool_host = mla_host
        radix_cache.token_to_kv_pool_host = host_pool_group
        radix_cache.cache_controller = cache_controller
        logger.info(
            "Hybrid hierarchical cache: HostPoolGroup(KV + INDEXER), HybridCacheController, "
            "transfer_layer_num=%s",
            layer_num,
        )
    except Exception:
        logger.exception("build_nsa_hybrid_stack failed")
        raise


def build_mamba_hybrid_stack(
    mamba_cache: "HiMambaRadixCache",
    params: "CacheInitParams",
    server_args: "ServerArgs",
    *,
    extra_config: dict,
    prefetch_threshold: int,
    load_cache_event,
    enable_storage_metrics: bool = False,
) -> None:
    """HostPoolGroup (KV + Mamba) + HybridCacheController for hybrid SSM models."""
    try:
        hybrid_kv = mamba_cache.hybrid_kv_cache
        kvcache = mamba_cache.kvcache
        kv_host_pool_cls = (
            MLATokenToKVPoolHost if hybrid_kv.use_mla else MHATokenToKVPoolHost
        )
        full_kv_pool_host = kv_host_pool_cls(
            kvcache,
            server_args.hicache_ratio,
            server_args.hicache_size,
            params.page_size,
            server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        mamba_pool_host = MambaPoolHost(
            params.req_to_token_pool.mamba_pool,
            server_args.hicache_ratio,
            server_args.hicache_size,
            allocator_type=server_args.hicache_storage_backend,
            layout=server_args.hicache_mem_layout,
        )

        full_layer_ids = sorted(hybrid_kv.full_attention_layer_id_mapping.keys())
        mamba_layer_ids = sorted(params.req_to_token_pool.mamba_map.keys())
        transfer_layer_num = len(set(full_layer_ids) | set(mamba_layer_ids))
        full_layer_mapping = dict(hybrid_kv.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)

        def kv_layer_mapper(layer_id: int) -> Optional[int]:
            if not 0 <= layer_id < transfer_layer_num:
                return None
            return full_layer_mapping.get(layer_id)

        def mamba_layer_mapper(layer_id: int) -> Optional[int]:
            if not 0 <= layer_id < transfer_layer_num:
                return None
            return mamba_layer_mapping.get(layer_id)

        host_pool_group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=full_kv_pool_host,
                    device_pool=kvcache,
                    layer_mapper=kv_layer_mapper,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name=PoolName.MAMBA,
                    host_pool=mamba_pool_host,
                    device_pool=params.req_to_token_pool.mamba_pool,
                    layer_mapper=mamba_layer_mapper,
                    host_evict_fn=mamba_cache.evict_mamba_host,
                    device_evict_fn=mamba_cache.evict_mamba,
                ),
            ]
        )
        cache_controller = HybridCacheController(
            params.token_to_kv_pool_allocator,
            host_pool_group,
            params.page_size,
            params.tp_cache_group,
            load_cache_event=load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            transfer_layer_num=transfer_layer_num,
            enable_storage_metrics=enable_storage_metrics,
        )
        mamba_cache.full_kv_pool_host = full_kv_pool_host
        mamba_cache.mamba_pool_host = mamba_pool_host
        mamba_cache.transfer_layer_num = transfer_layer_num
        mamba_cache.host_pool_group = host_pool_group
        mamba_cache.cache_controller = cache_controller
        params.req_to_token_pool.register_layer_transfer_counter(
            cache_controller.layer_done_counter
        )
        hybrid_kv.register_layer_transfer_counter(cache_controller.layer_done_counter)
        logger.info(
            "Hybrid hierarchical cache: HostPoolGroup(KV + MAMBA), HybridCacheController, "
            "transfer_layer_num=%s",
            transfer_layer_num,
        )
    except Exception:
        logger.exception("build_mamba_hybrid_stack failed")
        raise
