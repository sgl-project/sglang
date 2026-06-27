from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
    DSAIndexerPoolHost,
    HostPoolGroup,
    LogicalHostPool,
    MambaPoolHost,
    MLATokenToKVPoolHost,
    PoolEntry,
    get_mha_host_pool_cls,
)
from sglang.srt.mem_cache.unified_cache_components import ComponentType

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
    kv_host_pool_cls = (
        MLATokenToKVPoolHost if use_mla else get_mha_host_pool_cls(kv_pool)
    )
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
    host_evict_fn: Optional[Callable[[int], Any]] = None,
    device_evict_fn: Optional[Callable[[int], Any]] = None,
    device_alloc_fn: Optional[Callable[[int], Any]] = None,
    device_free_fn: Optional[Callable[[Any], Any]] = None,
) -> PoolEntry:
    return PoolEntry(
        name=name,
        host_pool=host_pool,
        device_pool=device_pool,
        layer_mapper=_make_layer_mapper(layer_mapping, transfer_layer_num),
        is_primary_index_anchor=is_anchor,
        host_evict_fn=host_evict_fn,
        device_evict_fn=device_evict_fn,
        device_alloc_fn=device_alloc_fn,
        device_free_fn=device_free_fn,
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
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
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
        pp_group=pp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_hybrid_swa_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    full_kv_pool: Any,
    swa_kv_pool: Any,
    full_layer_mapping: dict[int, int],
    swa_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping | swa_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=full_kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
    )
    swa_host_pool = build_kv_host_pool(
        kv_pool=swa_kv_pool,
        page_size=page_size,
        server_args=server_args,
        use_mla=use_mla,
    )

    # For SWA hybrid, the device alloc/free goes through the inner swa_attn_allocator
    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=full_kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.SWA,
            host_pool=swa_host_pool,
            device_pool=swa_kv_pool,
            layer_mapping=swa_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_swa_evict_fn,
            device_evict_fn=device_swa_evict_fn,
            device_alloc_fn=swa_attn_allocator.alloc,
            device_free_fn=swa_attn_allocator.free,
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
        pp_group=pp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def _deepseek_v4_num_host_pages(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kvcache: Any,
    page_size: int,
    swa_page_size: int,
) -> tuple[int, int]:
    allocator = params.token_to_kv_pool_allocator
    device_full_size = getattr(allocator, "size_full", kvcache.size)
    device_full_pages = (device_full_size + page_size - 1) // page_size

    device_swa_pages = (kvcache.swa_size + swa_page_size - 1) // swa_page_size

    if server_args.hicache_size > 0:
        raise ValueError(
            "DeepSeek V4 HiCache currently does not support --hicache-size; "
            "use --hicache-ratio instead."
        )
    ratio = server_args.hicache_ratio
    full_host_pages = int(device_full_pages * ratio)
    swa_host_pages = int(device_swa_pages * ratio)
    return full_host_pages, swa_host_pages


def build_deepseek_v4_hicache_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kvcache: Any,
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = kvcache.end_layer - kvcache.start_layer
    full_layer_mapping = {layer_id: layer_id for layer_id in range(transfer_layer_num)}
    if len(kvcache.swa_kv_pool.kv_buffer) != transfer_layer_num:
        raise ValueError(
            "DeepSeek V4 SWA KV pool must be PP-stage-local: "
            f"got {len(kvcache.swa_kv_pool.kv_buffer)} buffers for "
            f"{transfer_layer_num} local layers"
        )
    swa_layer_mapping = {layer_id: layer_id for layer_id in range(transfer_layer_num)}

    c4_layer_mapping = {}
    c128_layer_mapping = {}
    c4_state_local_layers = []
    c4_state_global_layers = []
    for local_layer_id, layer_item in enumerate(
        kvcache.layer_mapping[kvcache.start_layer : kvcache.end_layer]
    ):
        global_layer_id = kvcache.start_layer + local_layer_id
        if layer_item.compress_ratio == 4:
            c4_layer_mapping[local_layer_id] = layer_item.compress_layer_id
            c4_state_local_layers.append(local_layer_id)
            c4_state_global_layers.append(global_layer_id)
        elif layer_item.compress_ratio == 128:
            c128_layer_mapping[local_layer_id] = layer_item.compress_layer_id

    c4_state_mapping = {
        layer_id: local_id for local_id, layer_id in enumerate(c4_state_local_layers)
    }
    num_host_pages, swa_num_host_pages = _deepseek_v4_num_host_pages(
        params=params,
        server_args=server_args,
        kvcache=kvcache,
        page_size=page_size,
        swa_page_size=kvcache.swa_page_size,
    )

    logical_host_pool = LogicalHostPool(
        num_host_pages * page_size, page_size, layout=server_args.hicache_mem_layout
    )
    swa_host_pool = DeepSeekV4PagedHostPool(
        pool_name=str(PoolName.SWA),
        device_buffers=kvcache.swa_kv_pool.kv_buffer,
        item_bytes=kvcache.swa_kv_pool.bytes_per_page_padded,
        num_host_pages=swa_num_host_pages,
        slot_page_size=kvcache.swa_page_size,
        layout=server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )
    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=logical_host_pool,
            device_pool=kvcache,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.SWA,
            host_pool=swa_host_pool,
            device_pool=kvcache.swa_kv_pool,
            layer_mapping=swa_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_swa_evict_fn,
            device_evict_fn=device_swa_evict_fn,
            device_alloc_fn=swa_attn_allocator.alloc,
            device_free_fn=swa_attn_allocator.free,
        ),
    ]

    if c4_layer_mapping:
        c4_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4),
            device_buffers=kvcache.c4_kv_pool.kv_buffer,
            item_bytes=kvcache.c4_kv_pool.bytes_per_page_padded,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout=server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        c4_indexer_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER),
            device_buffers=kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer,
            item_bytes=(
                kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].shape[1]
                * kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].element_size()
            ),
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout=server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        c4_state_host_pool = DeepSeekV4StateHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4_STATE),
            state_pools=[
                kvcache.compress_state_pools[layer_id]
                for layer_id in c4_state_global_layers
            ],
            num_host_pages=swa_num_host_pages,
            swa_page_size=kvcache.swa_page_size,
            layout=server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        c4_indexer_state_host_pool = DeepSeekV4StateHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER_STATE),
            state_pools=[
                kvcache.indexer_compress_state_pools[layer_id]
                for layer_id in c4_state_global_layers
            ],
            num_host_pages=swa_num_host_pages,
            swa_page_size=kvcache.swa_page_size,
            layout=server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        entries.extend(
            [
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4,
                    host_pool=c4_host_pool,
                    device_pool=kvcache.c4_kv_pool,
                    layer_mapping=c4_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4_INDEXER,
                    host_pool=c4_indexer_host_pool,
                    device_pool=kvcache.c4_indexer_kv_pool,
                    layer_mapping=c4_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4_STATE,
                    host_pool=c4_state_host_pool,
                    device_pool=None,
                    layer_mapping=c4_state_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                    host_pool=c4_indexer_state_host_pool,
                    device_pool=None,
                    layer_mapping=c4_state_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
            ]
        )

    if c128_layer_mapping:
        c128_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C128),
            device_buffers=kvcache.c128_kv_pool.kv_buffer,
            item_bytes=kvcache.c128_kv_pool.bytes_per_page_padded,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout=server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )
        # C128 state pool is intentionally not registered with hicache.
        # page_size=256 % 128 == 0, so state pool is not consumed on load.
        entries.extend(
            [
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C128,
                    host_pool=c128_host_pool,
                    device_pool=kvcache.c128_kv_pool,
                    layer_mapping=c128_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                ),
            ]
        )

    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        pp_group=pp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
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
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    host_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    device_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
    mamba_allocator = params.req_to_token_pool.mamba_allocator
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
            device_alloc_fn=mamba_allocator.alloc,
            device_free_fn=mamba_allocator.free,
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
        pp_group=pp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def build_anchor_sidecar_stack(
    *,
    params: CacheInitParams,
    server_args: ServerArgs,
    kv_pool: Any,
    sidecar_pool_name: PoolName,
    full_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    sidecar_host_pool_factory: Callable[[Any], Any],
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
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
    sidecar_host_pool = sidecar_host_pool_factory(kv_host_pool)
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
            name=sidecar_pool_name,
            host_pool=sidecar_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
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
        pp_group=pp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


_COMPONENT_HOST_ATTR: dict[ComponentType, tuple[str, str]] = {
    ComponentType.FULL: ("full_kv_pool_host", "_full_kv_pool_host"),
    ComponentType.SWA: ("swa_kv_pool_host", "_swa_kv_pool_host"),
    ComponentType.MAMBA: ("mamba_pool_host", "_mamba_pool_host"),
}


@dataclass
class StackBuildResult:
    host_pool_group: HostPoolGroup
    cache_controller: HybridCacheController
    component_host_pools: dict[ComponentType, Any]
    sidecars: list[SidecarPoolSpec] = field(default_factory=list)
    # Mamba state lives in req_to_token_pool, not in kvcache, so its
    # layer_transfer_counter has to be wired separately.
    register_req_to_token_counter: bool = False
    transfer_layer_num: int = 0
    pools_desc: str = ""


class StackStrategy:
    def matches(self, kvcache: Any, components: set[ComponentType]) -> bool:
        raise NotImplementedError

    def build(
        self,
        *,
        cache: UnifiedRadixCache,
        kvcache: Any,
        params: CacheInitParams,
        server_args: ServerArgs,
        load_cache_event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        storage_backend: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        enable_storage_metrics: bool = False,
    ) -> StackBuildResult:
        raise NotImplementedError


class _DeepSeekV4Strategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        return isinstance(kvcache, DeepSeekV4TokenToKVPool) and components == {
            ComponentType.FULL,
            ComponentType.SWA,
        }

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        attn_cp_group=None,
        attn_tp_group=None,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        host_pool_group, cache_controller = build_deepseek_v4_hicache_stack(
            params=params,
            server_args=server_args,
            kvcache=kvcache,
            page_size=cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=params.pp_cache_group,
            storage_backend=storage_backend,
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
            device_swa_evict_fn=lambda n: cache.evict(EvictParams(swa_num_tokens=n)),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        sidecars = [
            SidecarPoolSpec(
                pool_name=name,
                indices_from_pool=src,
                hit_policy=(
                    PoolHitPolicy.TRAILING_PAGES
                    if src == PoolName.SWA
                    else PoolHitPolicy.ALL_PAGES
                ),
            )
            for name, src in (
                (PoolName.DEEPSEEK_V4_C4, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C4_INDEXER, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C128, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C4_STATE, PoolName.SWA),
                (PoolName.DEEPSEEK_V4_C4_INDEXER_STATE, PoolName.SWA),
                (PoolName.DEEPSEEK_V4_C128_STATE, PoolName.SWA),
            )
            if name in host_pool_group.entry_map
        ]
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
                ComponentType.SWA: host_pool_group.get_pool(PoolName.SWA),
            },
            sidecars=sidecars,
            transfer_layer_num=kvcache.end_layer - kvcache.start_layer,
            pools_desc="KV + SWA + DeepSeekV4 sidecars",
        )


class _MambaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        return isinstance(kvcache, HybridLinearKVPool) and components == {
            ComponentType.FULL,
            ComponentType.MAMBA,
        }

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        attn_cp_group=None,
        attn_tp_group=None,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        full_layer_mapping = dict(kvcache.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
        host_pool_group, cache_controller = build_hybrid_mamba_stack(
            params=params,
            server_args=server_args,
            kv_pool=kvcache.full_kv_pool,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping=full_layer_mapping,
            mamba_layer_mapping=mamba_layer_mapping,
            page_size=cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=params.pp_cache_group,
            storage_backend=storage_backend,
            use_mla=kvcache.use_mla,
            host_mamba_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA),
            device_mamba_evict_fn=lambda n: cache.evict(EvictParams(mamba_num=n)),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
                ComponentType.MAMBA: host_pool_group.get_pool(PoolName.MAMBA),
            },
            register_req_to_token_counter=True,
            transfer_layer_num=len(full_layer_mapping | mamba_layer_mapping),
            pools_desc="KV + MAMBA",
        )


def _swa_layer_mappings(kvcache) -> tuple[dict[int, int], dict[int, int]]:
    full = {
        gid: lid for gid, (lid, is_swa) in kvcache.layers_mapping.items() if not is_swa
    }
    swa = {gid: lid for gid, (lid, is_swa) in kvcache.layers_mapping.items() if is_swa}
    return full, swa


class _SwaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        return (
            isinstance(kvcache, SWAKVPool)
            and not isinstance(kvcache, DeepSeekV4TokenToKVPool)
            and components == {ComponentType.FULL, ComponentType.SWA}
        )

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        attn_cp_group=None,
        attn_tp_group=None,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        full_layer_mapping, swa_layer_mapping = _swa_layer_mappings(kvcache)
        host_pool_group, cache_controller = build_hybrid_swa_stack(
            params=params,
            server_args=server_args,
            full_kv_pool=kvcache.full_kv_pool,
            swa_kv_pool=kvcache.swa_kv_pool,
            full_layer_mapping=full_layer_mapping,
            swa_layer_mapping=swa_layer_mapping,
            page_size=cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=params.pp_cache_group,
            storage_backend=storage_backend,
            use_mla=False,
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
            device_swa_evict_fn=lambda n: cache.evict(EvictParams(swa_num_tokens=n)),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
                ComponentType.SWA: host_pool_group.get_pool(PoolName.SWA),
            },
            transfer_layer_num=len(full_layer_mapping | swa_layer_mapping),
            pools_desc="KV + SWA",
        )


class _DsaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        return isinstance(kvcache, DSATokenToKVPool) and components == {
            ComponentType.FULL
        }

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        attn_cp_group=None,
        attn_tp_group=None,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        full_kv_pool = kvcache
        use_mla = isinstance(kvcache, MLATokenToKVPool)
        full_layer_mapping = {i: i for i in range(full_kv_pool.layer_num)}
        host_pool_group, cache_controller = build_anchor_sidecar_stack(
            params=params,
            server_args=server_args,
            kv_pool=full_kv_pool,
            sidecar_pool_name=PoolName.INDEXER,
            full_layer_mapping=full_layer_mapping,
            page_size=cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            storage_backend=storage_backend,
            use_mla=use_mla,
            override_kv_cache_dim=full_kv_pool.kv_cache_dim,
            sidecar_host_pool_factory=lambda kv_host_pool: DSAIndexerPoolHost(
                full_kv_pool,
                kv_host_pool,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            ),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
            },
            sidecars=[
                SidecarPoolSpec(
                    pool_name=PoolName.INDEXER,
                    indices_from_pool=PoolName.KV,
                ),
            ],
            transfer_layer_num=len(full_layer_mapping),
            pools_desc="KV + INDEXER",
        )


class _PlainKvStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.memory_pool import (
            DSATokenToKVPool,
            HybridLinearKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        if isinstance(
            kvcache,
            (SWAKVPool, HybridLinearKVPool, DSATokenToKVPool, DeepSeekV4TokenToKVPool),
        ):
            return False
        return components == {ComponentType.FULL}

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        attn_cp_group=None,
        attn_tp_group=None,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        full_kv_pool = kvcache
        use_mla = isinstance(kvcache, MLATokenToKVPool)
        full_layer_mapping = {i: i for i in range(full_kv_pool.layer_num)}
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
            pp_group=params.pp_cache_group,
            storage_backend=storage_backend,
            use_mla=use_mla,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
            },
            transfer_layer_num=len(full_layer_mapping),
            pools_desc="KV",
        )


# Resolved first-to-last; _PlainKvStrategy is the catch-all fallback.
_STRATEGIES: list[StackStrategy] = [
    _DeepSeekV4Strategy(),
    _MambaStrategy(),
    _SwaStrategy(),
    _DsaStrategy(),
    _PlainKvStrategy(),
]


def register_stack_strategy(strategy: StackStrategy) -> None:
    """Prepend a strategy so downstream forks can plug in (kvcache, components)
    combinations not in the built-in list."""
    _STRATEGIES.insert(0, strategy)


def _select_strategy(kvcache: Any, components: set[ComponentType]) -> StackStrategy:
    for strategy in _STRATEGIES:
        if strategy.matches(kvcache, components):
            return strategy
    raise AssertionError(
        f"No matching HiCache strategy for kvcache={type(kvcache).__name__}, "
        f"components={sorted(c.name for c in components)}"
    )


def _apply_stack_result(
    cache: UnifiedRadixCache,
    kvcache: Any,
    params: CacheInitParams,
    result: StackBuildResult,
) -> None:
    cache.host_pool_group = result.host_pool_group
    cache.cache_controller = result.cache_controller

    for ct, host_pool in result.component_host_pools.items():
        cache_attr, component_attr = _COMPONENT_HOST_ATTR[ct]
        setattr(cache, cache_attr, host_pool)
        setattr(cache.components[ct], component_attr, host_pool)

    for sidecar in result.sidecars:
        cache.register_sidecar_pool(sidecar)

    kvcache.register_layer_transfer_counter(result.cache_controller.layer_done_counter)
    if result.register_req_to_token_counter:
        params.req_to_token_pool.register_layer_transfer_counter(
            result.cache_controller.layer_done_counter
        )

    logger.info(
        "Attached hybrid pool stack to UnifiedRadixCache: pools=%s, transfer_layer_num=%s",
        result.pools_desc,
        result.transfer_layer_num,
    )


def attach_hybrid_pool_to_unified_cache(
    cache: UnifiedRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str] = None,
    storage_extra_config: Optional[dict] = None,
    storage_prefetch_threshold: int = 256,
) -> None:
    """Attach HostPoolGroup + HybridCacheController to UnifiedRadixCache."""
    try:
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        components = set(cache.components.keys())
        strategy = _select_strategy(kvcache, components)
        result = strategy.build(
            cache=cache,
            kvcache=kvcache,
            params=params,
            server_args=server_args,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            storage_backend=storage_backend,
            storage_backend_extra_config=storage_extra_config,
            prefetch_threshold=storage_prefetch_threshold,
            model_name=server_args.served_model_name,
            enable_storage_metrics=cache._enable_metrics_flag,
        )
        _apply_stack_result(cache, kvcache, params, result)
    except Exception:
        logger.exception("attach_hybrid_pool_to_unified_cache failed")
        raise


def attach_hybrid_dsa_pool_to_hiradix_cache(
    radix_cache: HiRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Attach HostPoolGroup (KV + indexer) + HybridCacheController for HiRadixCache.

    This entrypoint is currently intended only for HiRadixCache's DSA path.
    """
    try:
        kv = radix_cache.kv_cache
        layer_mapping = {layer_id: layer_id for layer_id in range(kv.layer_num)}
        host_pool_group, cache_controller = build_anchor_sidecar_stack(
            params=params,
            server_args=server_args,
            kv_pool=kv,
            sidecar_pool_name=PoolName.INDEXER,
            full_layer_mapping=layer_mapping,
            page_size=radix_cache.page_size,
            tp_group=radix_cache.tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=radix_cache.pp_group,
            storage_backend=server_args.hicache_storage_backend,
            use_mla=True,
            override_kv_cache_dim=kv.kv_cache_dim,
            prefetch_threshold=prefetch_threshold,
            sidecar_host_pool_factory=lambda kv_host_pool: DSAIndexerPoolHost(
                kv,
                kv_host_pool,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            ),
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        radix_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
        radix_cache.token_to_kv_pool_host = host_pool_group
        radix_cache.cache_controller = cache_controller
        logger.info(
            "Attached hybrid DSA pool stack to HiRadixCache: pools=KV + INDEXER, "
            "transfer_layer_num=%s",
            len(layer_mapping),
        )
    except Exception:
        logger.exception("attach_hybrid_dsa_pool_to_hiradix_cache failed")
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
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
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
            pp_group=params.pp_cache_group,
            storage_backend=server_args.hicache_storage_backend,
            use_mla=hybrid_kv.use_mla,
            host_mamba_evict_fn=mamba_cache.evict_mamba_host,
            device_mamba_evict_fn=mamba_cache.evict_mamba,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
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
