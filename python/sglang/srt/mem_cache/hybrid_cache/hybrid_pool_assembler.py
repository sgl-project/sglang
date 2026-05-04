from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName
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
    import torch

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _make_layer_mapper(
    layer_mapping: dict[int, int],
    transfer_layer_num: int,
) -> Callable[[int], Optional[int]]:
    def mapper(layer_id: int) -> Optional[int]:
        if not 0 <= layer_id < transfer_layer_num:
            return None
        return layer_mapping.get(layer_id)

    return mapper


def build_kv_host_pool(
    *,
    kv_pool: Any,
    page_size: int,
    server_args: ServerArgs,
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
):
    kv_host_pool_cls = MLATokenToKVPoolHost if use_mla else MHATokenToKVPoolHost
    kwargs = {}
    if override_kv_cache_dim is not None:
        kwargs["override_kv_cache_dim"] = override_kv_cache_dim
    return kv_host_pool_cls(
        kv_pool,
        server_args.hicache_ratio,
        server_args.hicache_size,
        page_size,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
        **kwargs,
    )


def build_pool_entry(
    *,
    name: PoolName,
    host_pool: Any,
    device_pool: Any,
    layer_mapping: dict[int, int],
    transfer_layer_num: int,
    is_anchor: bool = False,
    share_indices_with_anchor: bool = False,
    host_evict_fn: Optional[Callable[[int], Any]] = None,
    device_evict_fn: Optional[Callable[[int], Any]] = None,
) -> PoolEntry:
    return PoolEntry(
        name=name,
        host_pool=host_pool,
        device_pool=device_pool,
        layer_mapper=_make_layer_mapper(layer_mapping, transfer_layer_num),
        is_primary_index_anchor=is_anchor,
        share_indices_with_anchor=share_indices_with_anchor,
        host_evict_fn=host_evict_fn,
        device_evict_fn=device_evict_fn,
    )


def build_kv_only_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    full_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        )
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_hybrid_mamba_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    mamba_pool: Any,
    full_layer_mapping: dict[int, int],
    mamba_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    host_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    device_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
    )
    mamba_host_pool = MambaPoolHost(
        mamba_pool,
        server_args.hicache_ratio,
        server_args.hicache_size,
        allocator_type=server_args.hicache_storage_backend,
        layout=server_args.hicache_mem_layout,
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.MAMBA,
            host_pool=mamba_host_pool,
            device_pool=mamba_pool,
            layer_mapping=mamba_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_mamba_evict_fn,
            device_evict_fn=device_mamba_evict_fn,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_shared_anchor_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    shared_pool_name: PoolName,
    full_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    shared_host_pool_factory: Callable[[Any], Any],
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
    )
    shared_host_pool = shared_host_pool_factory(kv_host_pool)
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=shared_pool_name,
            host_pool=shared_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            share_indices_with_anchor=True,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def attach_hybrid_pool_to_unified_cache(
    cache: UnifiedRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    load_cache_event,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> None:
    """Attach HostPoolGroup + HybridCacheController to UnifiedRadixCache."""
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        MLATokenToKVPool,
        NSATokenToKVPool,
    )
    from sglang.srt.mem_cache.unified_cache_components import ComponentType

    try:
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kvcache, HybridLinearKVPool):
            full_kv_pool = kvcache.full_kv_pool
            use_mla = kvcache.use_mla
            assert set(cache.components.keys()) == {
                ComponentType.FULL,
                ComponentType.MAMBA,
            }, "HybridLinearKVPool currently only supports FULL + MAMBA in UnifiedRadixCache."
        else:
            full_kv_pool = kvcache
            use_mla = isinstance(kvcache, MLATokenToKVPool)
            assert set(cache.components.keys()) == {
                ComponentType.FULL
            }, "Non-hybrid KV pool currently only supports FULL-only UnifiedRadixCache."

        mamba_stack = isinstance(kvcache, HybridLinearKVPool)
        nsa_stack = isinstance(kvcache, NSATokenToKVPool)
        if mamba_stack:
            full_layer_mapping = dict(kvcache.full_attention_layer_id_mapping)
            mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
            host_pool_group, cache_controller = build_hybrid_mamba_stack(
                params=params,
                server_args=server_args,
                kv_pool=full_kv_pool,
                mamba_pool=params.req_to_token_pool.mamba_pool,
                full_layer_mapping=full_layer_mapping,
                mamba_layer_mapping=mamba_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                use_mla=use_mla,
                host_mamba_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA),
                device_mamba_evict_fn=lambda n: cache.evict(EvictParams(mamba_num=n)),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            cache.mamba_pool_host = host_pool_group.get_pool(PoolName.MAMBA)
            cache.components[ComponentType.MAMBA]._mamba_pool_host = (
                cache.mamba_pool_host
            )
            params.req_to_token_pool.register_layer_transfer_counter(
                cache_controller.layer_done_counter
            )
            transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
        elif nsa_stack:
            full_layer_mapping = {
                layer_id: layer_id for layer_id in range(full_kv_pool.layer_num)
            }
            host_pool_group, cache_controller = build_shared_anchor_stack(
                params=params,
                server_args=server_args,
                kv_pool=full_kv_pool,
                shared_pool_name=PoolName.INDEXER,
                full_layer_mapping=full_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                storage_backend=None,
                use_mla=use_mla,
                shared_host_pool_factory=lambda kv_host_pool: NSAIndexerPoolHost(
                    full_kv_pool,
                    kv_host_pool,
                    server_args.hicache_mem_layout,
                    allocator_type=server_args.hicache_storage_backend,
                ),
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
                attn_cp_rank=params.attn_cp_rank,
                attn_cp_size=params.attn_cp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            # Register the NSA indexer pool as sharing anchor-KV indices so
            # HiCache backup/load emits its PoolTransfer together with KV.
            cache.register_hicache_anchor_kv_shared_indices_pool(
                PoolName.INDEXER,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            )
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            transfer_layer_num = len(full_layer_mapping)
        else:
            full_layer_mapping = {
                layer_id: layer_id for layer_id in range(full_kv_pool.layer_num)
            }
            host_pool_group, cache_controller = build_kv_only_stack(
                params=params,
                server_args=server_args,
                kv_pool=full_kv_pool,
                full_layer_mapping=full_layer_mapping,
                page_size=cache.page_size,
                tp_group=params.tp_cache_group,
                load_cache_event=load_cache_event,
                attn_cp_group=attn_cp_group,
                attn_tp_group=attn_tp_group,
                storage_backend=None,
                use_mla=use_mla,
                pp_rank=params.pp_rank,
                pp_size=params.pp_size,
            )
            cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
            cache.host_pool_group = host_pool_group
            cache.cache_controller = cache_controller
            cache.components[ComponentType.FULL]._full_kv_pool_host = (
                cache.full_kv_pool_host
            )
            transfer_layer_num = len(full_layer_mapping)

        kvcache.register_layer_transfer_counter(
            cache.cache_controller.layer_done_counter
        )

        logger.info(
            "Attached hybrid pool stack to UnifiedRadixCache: pools=%s, transfer_layer_num=%s",
            "KV + MAMBA" if mamba_stack else "KV + INDEXER" if nsa_stack else "KV",
            transfer_layer_num,
        )
    except Exception:
        logger.exception("attach_hybrid_pool_to_unified_cache failed")
        raise


def attach_hybrid_nsa_pool_to_hiradix_cache(
    radix_cache: HiRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> None:
    """Attach HostPoolGroup (KV + indexer) + HybridCacheController for HiRadixCache.

    This entrypoint is currently intended only for HiRadixCache's NSA path.
    """
    try:
        kv = radix_cache.kv_cache
        layer_mapping = {layer_id: layer_id for layer_id in range(kv.layer_num)}
        host_pool_group, cache_controller = build_shared_anchor_stack(
            params=params,
            server_args=server_args,
            kv_pool=kv,
            shared_pool_name=PoolName.INDEXER,
            full_layer_mapping=layer_mapping,
            page_size=radix_cache.page_size,
            tp_group=radix_cache.tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            storage_backend=server_args.hicache_storage_backend,
            use_mla=True,
            prefetch_threshold=prefetch_threshold,
            shared_host_pool_factory=lambda kv_host_pool: NSAIndexerPoolHost(
                kv,
                kv_host_pool,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            ),
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=radix_cache.pp_rank,
            pp_size=radix_cache.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            enable_storage_metrics=enable_storage_metrics,
        )
        radix_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
        radix_cache.token_to_kv_pool_host = host_pool_group
        radix_cache.cache_controller = cache_controller
        logger.info(
            "Attached hybrid NSA pool stack to HiRadixCache: pools=KV + INDEXER, "
            "transfer_layer_num=%s",
            len(layer_mapping),
        )
    except Exception:
        logger.exception("attach_hybrid_nsa_pool_to_hiradix_cache failed")
        raise


def attach_hybrid_pool_to_mamba_cache(
    mamba_cache: HiMambaRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    load_cache_event,
    enable_storage_metrics: bool = False,
    attn_cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    attn_tp_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> None:
    """Attach HostPoolGroup (KV + Mamba) + HybridCacheController for HiMambaRadixCache.

    This entrypoint is currently intended only for HiMambaRadixCache.
    """
    try:
        hybrid_kv = mamba_cache.hybrid_kv_cache
        kvcache = mamba_cache.kvcache
        full_layer_mapping = dict(hybrid_kv.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
        host_pool_group, cache_controller = build_hybrid_mamba_stack(
            params=params,
            server_args=server_args,
            kv_pool=kvcache,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping=full_layer_mapping,
            mamba_layer_mapping=mamba_layer_mapping,
            page_size=params.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            storage_backend=server_args.hicache_storage_backend,
            use_mla=hybrid_kv.use_mla,
            host_mamba_evict_fn=mamba_cache.evict_mamba_host,
            device_mamba_evict_fn=mamba_cache.evict_mamba,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            enable_storage_metrics=enable_storage_metrics,
        )
        mamba_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
        mamba_cache.mamba_pool_host = host_pool_group.get_pool(PoolName.MAMBA)
        mamba_cache.transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
        mamba_cache.host_pool_group = host_pool_group
        mamba_cache.cache_controller = cache_controller
        params.req_to_token_pool.register_layer_transfer_counter(
            cache_controller.layer_done_counter
        )
        hybrid_kv.register_layer_transfer_counter(cache_controller.layer_done_counter)
        logger.info(
            "Attached hybrid Mamba pool stack to HiMambaRadixCache: pools=KV + MAMBA, "
            "transfer_layer_num=%s",
            mamba_cache.transfer_layer_num,
        )
    except Exception:
        logger.exception("attach_hybrid_pool_to_mamba_cache failed")
        raise
