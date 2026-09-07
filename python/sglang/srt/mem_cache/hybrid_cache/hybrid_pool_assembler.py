from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

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
    LogicalHostPool,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.common import get_allocator_type
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKOnlyPoolHost,
    get_mha_host_pool_cls,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import (
    get_memory,
    get_parallel,
    get_serving,
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _get_allocator_type() -> str:
    return get_allocator_type()


def _evict_swa_for_device_alloc(cache: UnifiedRadixCache, required_size: int) -> None:
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    available_size = cache.token_to_kv_pool_allocator.swa_available_size()
    shortfall = max(0, required_size - available_size)
    if shortfall > 0:
        cache.evict_for_alloc(EvictParams(swa_num_tokens=shortfall))


def _evict_mamba_for_device_alloc(cache: UnifiedRadixCache, required_size: int) -> None:
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    available_size = (
        cache.req_to_token_pool.mamba_allocator.schedulable_available_size()
    )
    shortfall = max(0, required_size - available_size)
    if shortfall > 0:
        cache.evict_for_alloc(EvictParams(mamba_num=shortfall))


def _make_layer_mapper(
    layer_mapping: dict[int, int],
    transfer_layer_num: int,
) -> Callable[[int], Optional[int]]:
    def mapper(layer_id: int) -> Optional[int]:
        if not 0 <= layer_id < transfer_layer_num:
            return None
        return layer_mapping.get(layer_id)

    return mapper


def _with_mtp_layer_mapping(
    layer_mapping: dict[int, int],
    *,
    transfer_layer_start: int,
    target_device_layer_num: int,
    draft_layer_num: int,
) -> dict[int, int]:
    return layer_mapping | {
        transfer_layer_start + depth: target_device_layer_num + depth
        for depth in range(draft_layer_num)
    }


class _DeepSeekV4LayerMappings(NamedTuple):
    transfer_layer_num: int
    full: dict[int, int]
    swa: dict[int, int]
    c4: dict[int, int]
    c128: dict[int, int]
    c4_state: dict[int, int]
    c4_state_global_layers: list[int]


def _resolve_deepseek_v4_layer_mappings(
    kvcache: Any,
) -> _DeepSeekV4LayerMappings:
    transfer_layer_num = kvcache.end_layer - kvcache.start_layer
    full = {layer: layer for layer in range(transfer_layer_num)}
    swa = {} if getattr(kvcache, "_unified_kv", False) else full.copy()

    c4, c128, c4_state_global_layers = {}, {}, []
    for local_layer, item in enumerate(
        kvcache.layer_mapping[kvcache.start_layer : kvcache.end_layer]
    ):
        if item.compress_ratio == 4:
            c4[local_layer] = item.compress_layer_id
            c4_state_global_layers.append(kvcache.start_layer + local_layer)
        elif item.compress_ratio == 128:
            c128[local_layer] = item.compress_layer_id

    return _DeepSeekV4LayerMappings(
        transfer_layer_num=transfer_layer_num,
        full=full,
        swa=swa,
        c4=c4,
        c128=c128,
        c4_state={layer: index for index, layer in enumerate(c4)},
        c4_state_global_layers=c4_state_global_layers,
    )


def build_kv_host_pool(
    *,
    kv_pool: Any,
    page_size: int,
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    host_size: Optional[float] = None,
    mtp_draft_device_pools: tuple[Any, ...] = (),
    pool_label: str = "kv",
):
    kv_host_pool_cls = (
        MLATokenToKVPoolHost if use_mla else get_mha_host_pool_cls(kv_pool)
    )
    kwargs = {}
    if override_kv_cache_dim is not None:
        kwargs["override_kv_cache_dim"] = override_kv_cache_dim
    if mtp_draft_device_pools:
        kwargs["mtp_draft_device_pools"] = mtp_draft_device_pools
    parallel = get_parallel()
    if parallel.dcp_enabled:
        assert use_mla, (
            "HiCache + DCP is only wired for the MLA host pool; the MHA host "
            "pool has no DCP index translation."
        )
        kwargs["dcp_size"] = parallel.attn_dcp_size
        kwargs["dcp_rank"] = parallel.attn_dcp_rank
    return kv_host_pool_cls(
        kv_pool,
        get_memory().hicache_ratio,
        get_memory().hicache_size if host_size is None else host_size,
        page_size,
        get_memory().hicache_mem_layout,
        allocator_type=_get_allocator_type(),
        pool_label=pool_label,
        **kwargs,
    )


def _split_hicache_size(
    hicache_size: int, kv_pools: tuple[Any, ...]
) -> tuple[float, ...]:
    device_pool_sizes = []
    for kv_pool in kv_pools:
        size_bytes = kv_pool.get_kv_size_bytes()
        device_pool_sizes.append(
            sum(size_bytes) if isinstance(size_bytes, tuple) else size_bytes
        )
    total_device_pool_size = sum(device_pool_sizes)
    return tuple(
        hicache_size * size_bytes / total_device_pool_size
        for size_bytes in device_pool_sizes
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
    packed_draft_device_pools: tuple[Any, ...] = (),
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
        packed_draft_device_pools=packed_draft_device_pools,
    )


def build_kv_only_group(
    *,
    page_size: int,
    kv_pool: Any,
    full_layer_mapping: dict[int, int],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    host_size: Optional[float] = None,
    mtp_draft_device_pools: tuple[Any, ...] = (),
) -> HostPoolGroup:
    """Anchor-only host pool group for a flat MHA/MLA device pool."""
    transfer_layer_num = len(full_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
        host_size=host_size,
        mtp_draft_device_pools=mtp_draft_device_pools,
    )
    if mtp_draft_device_pools:
        full_layer_mapping = _with_mtp_layer_mapping(
            full_layer_mapping,
            transfer_layer_start=transfer_layer_num,
            target_device_layer_num=kv_pool.layer_num,
            draft_layer_num=len(mtp_draft_device_pools),
        )
    return HostPoolGroup(
        [
            build_pool_entry(
                name=PoolName.KV,
                host_pool=kv_host_pool,
                device_pool=kv_pool,
                layer_mapping=full_layer_mapping,
                transfer_layer_num=transfer_layer_num + len(mtp_draft_device_pools),
                is_anchor=True,
                packed_draft_device_pools=mtp_draft_device_pools,
            )
        ]
    )


def build_hybrid_swa_group(
    *,
    page_size: int,
    full_kv_pool: Any,
    swa_kv_pool: Any,
    full_layer_mapping: dict[int, int],
    swa_layer_mapping: dict[int, int],
    use_mla: bool,
    kv_host_size: Optional[float] = None,
    swa_host_size: Optional[float] = None,
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    swa_attn_allocator: Any = None,
    mtp_swa_device_pools: tuple[Any, ...] = (),
) -> HostPoolGroup:
    """Anchor (full) + SWA host pool group for a hybrid-SWA device pool."""
    transfer_layer_num = len(full_layer_mapping | swa_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=full_kv_pool,
        page_size=page_size,
        use_mla=use_mla,
        host_size=kv_host_size,
        pool_label="full",
    )
    swa_host_pool = build_kv_host_pool(
        kv_pool=swa_kv_pool,
        page_size=page_size,
        use_mla=use_mla,
        host_size=swa_host_size,
        mtp_draft_device_pools=mtp_swa_device_pools,
        pool_label="swa",
    )
    if mtp_swa_device_pools:
        swa_layer_mapping = _with_mtp_layer_mapping(
            swa_layer_mapping,
            transfer_layer_start=transfer_layer_num,
            target_device_layer_num=swa_kv_pool.layer_num,
            draft_layer_num=len(mtp_swa_device_pools),
        )
    return HostPoolGroup(
        [
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
                transfer_layer_num=transfer_layer_num + len(mtp_swa_device_pools),
                host_evict_fn=host_swa_evict_fn,
                device_evict_fn=device_swa_evict_fn,
                device_alloc_fn=(
                    swa_attn_allocator.alloc if swa_attn_allocator is not None else None
                ),
                device_free_fn=(
                    swa_attn_allocator.free if swa_attn_allocator is not None else None
                ),
                packed_draft_device_pools=mtp_swa_device_pools,
            ),
        ]
    )


def build_kv_only_stack(
    *,
    params: CacheInitParams,
    kv_pool: Any,
    full_layer_mapping: dict[int, int],
    load_cache_event,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping)
    host_pool_group = build_kv_only_group(
        page_size=params.page_size,
        kv_pool=kv_pool,
        full_layer_mapping=full_layer_mapping,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
        mtp_draft_device_pools=params.mtp_draft_device_pools,
    )
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        params.page_size,
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def build_hybrid_swa_stack(
    *,
    params: CacheInitParams,
    full_kv_pool: Any,
    swa_kv_pool: Any,
    full_layer_mapping: dict[int, int],
    swa_layer_mapping: dict[int, int],
    load_cache_event,
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
    # MTP draft pools follow the target SWA layout; select their SWA storage.
    mtp_swa_device_pools = tuple(
        pool.swa_kv_pool for pool in params.mtp_draft_device_pools
    )

    kv_host_size = swa_host_size = None
    if get_memory().hicache_size > 0:
        kv_host_size, swa_host_size = _split_hicache_size(
            get_memory().hicache_size, (full_kv_pool, swa_kv_pool)
        )

    host_pool_group = build_hybrid_swa_group(
        page_size=params.page_size,
        full_kv_pool=full_kv_pool,
        swa_kv_pool=swa_kv_pool,
        full_layer_mapping=full_layer_mapping,
        swa_layer_mapping=swa_layer_mapping,
        use_mla=use_mla,
        kv_host_size=kv_host_size,
        swa_host_size=swa_host_size,
        host_swa_evict_fn=host_swa_evict_fn,
        device_swa_evict_fn=device_swa_evict_fn,
        # For SWA hybrid, device allocation goes through the inner allocator.
        swa_attn_allocator=params.token_to_kv_pool_allocator.swa_attn_allocator,
        mtp_swa_device_pools=mtp_swa_device_pools,
    )
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        params.page_size,
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def _deepseek_v4_num_host_pages(
    *,
    params: CacheInitParams,
    kvcache: Any,
    page_size: int,
    swa_page_size: int,
) -> tuple[int, int]:
    allocator = params.token_to_kv_pool_allocator
    device_full_size = getattr(allocator, "size_full", kvcache.size)
    device_full_pages = (device_full_size + page_size - 1) // page_size

    device_swa_pages = (kvcache.swa_size + swa_page_size - 1) // swa_page_size

    if get_memory().hicache_size > 0:
        raise ValueError(
            "DeepSeek V4 HiCache currently does not support --hicache-size; "
            "use --hicache-ratio instead."
        )
    ratio = get_memory().hicache_ratio
    full_host_pages = int(device_full_pages * ratio)
    swa_host_pages = int(device_swa_pages * ratio)
    return full_host_pages, swa_host_pages


def _dsv4_compressed_region_buffers(kvcache: Any, ratio: int) -> tuple[list, int]:
    """
    Resolve ``(device_buffers, item_bytes)`` for a DeepSeek V4 C4/C128 main-KV
    HiCache pool, hiding the device KV layout from the stack builder.
    """
    if getattr(kvcache, "_unified_kv", False):
        return kvcache.unified_region_buffers(ratio)
    pool = kvcache.c4_kv_pool if ratio == 4 else kvcache.c128_kv_pool
    return pool.kv_buffer, pool.bytes_per_page_padded


@dataclass(frozen=True)
class _IndexerRegion:
    """One page-contiguous indexer buffer group to mirror on the host."""

    name: PoolName
    device_buffers: list
    item_bytes: int
    # FP4 page rows group their slots instead of laying tokens out flat, so the
    # fused-row token-granular copy does not apply and transfers must be whole
    # pages. The fused FP8 row has no such restriction.
    page_aligned_only: bool


def _dsv4_indexer_regions(kvcache: Any, page_size: int) -> list[_IndexerRegion]:
    """
    Resolve the indexer HiCache regions, hiding the FP8/FP4 split from the
    stack builder. FP8 keeps key and scale fused in one buffer, while FP4
    stores payload and scale separately, so it maps to two host pools.
    """
    import torch

    pool = kvcache.c4_indexer_kv_pool
    fused = pool.index_k_with_scale_buffer
    if fused is not None:
        return [
            _IndexerRegion(
                name=PoolName.DEEPSEEK_V4_C4_INDEXER,
                device_buffers=fused,
                item_bytes=fused[0].shape[1] * fused[0].element_size(),
                page_aligned_only=False,
            )
        ]

    payload_ref = pool.index_k_payload_buffer[0]
    scale_ref = pool.index_k_scale_buffer[0]
    # A page row covers ``page_slots`` C4 slots, i.e. one tree page of tokens
    # after 4:1 compression.
    page_slots = payload_ref.shape[3]
    if scale_ref.shape[3] != page_slots:
        raise ValueError(
            "FP4 indexer payload and scale must agree on slots per page: "
            f"payload={page_slots}, scale={scale_ref.shape[3]}"
        )
    if page_size % page_slots != 0:
        raise ValueError(
            f"Tree page size {page_size} must be a multiple of the FP4 indexer "
            f"slots per page {page_slots}"
        )
    payload = [b.view(torch.uint8).flatten(1) for b in pool.index_k_payload_buffer]
    scale = [b.view(torch.uint8).flatten(1) for b in pool.index_k_scale_buffer]
    return [
        _IndexerRegion(
            name=PoolName.DEEPSEEK_V4_C4_INDEXER,
            device_buffers=payload,
            item_bytes=payload[0].shape[1],
            page_aligned_only=True,
        ),
        _IndexerRegion(
            name=PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
            device_buffers=scale,
            item_bytes=scale[0].shape[1],
            page_aligned_only=True,
        ),
    ]


def build_deepseek_v4_hicache_stack(
    *,
    params: CacheInitParams,
    kvcache: Any,
    load_cache_event,
    storage_backend: Optional[str],
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
    layer_mappings: Optional[_DeepSeekV4LayerMappings] = None,
) -> tuple[HostPoolGroup, HybridCacheController]:
    page_size = params.page_size
    layer_mappings = layer_mappings or _resolve_deepseek_v4_layer_mappings(kvcache)
    transfer_layer_num = layer_mappings.transfer_layer_num
    full_layer_mapping = layer_mappings.full

    is_unified_kv = getattr(kvcache, "_unified_kv", False)
    mtp_swa_device_buffers = []
    if is_unified_kv:
        # unified_kv keeps the SWA ring inside the unified pool and never offloads it,
        # so there is no separate SWA host pool to map.
        swa_layer_mapping = {}
    else:
        if len(kvcache.swa_kv_pool.kv_buffer) != transfer_layer_num:
            raise ValueError(
                "DeepSeek V4 SWA KV pool must be PP-stage-local: "
                f"got {len(kvcache.swa_kv_pool.kv_buffer)} buffers for "
                f"{transfer_layer_num} local layers"
            )
        swa_layer_mapping = layer_mappings.swa
        # Keep every uncompressed draft SWA layer after the target SWA layers.
        # NextN has one layer per pool, while DSpark keeps all stages in one pool.
        mtp_swa_device_buffers = [
            buffer
            for pool in params.mtp_draft_device_pools
            for buffer in pool.swa_kv_pool.kv_buffer
        ]
        swa_layer_mapping = _with_mtp_layer_mapping(
            swa_layer_mapping,
            transfer_layer_start=transfer_layer_num,
            target_device_layer_num=transfer_layer_num,
            draft_layer_num=len(mtp_swa_device_buffers),
        )

    c4_layer_mapping = layer_mappings.c4
    c128_layer_mapping = layer_mappings.c128
    c4_state_mapping = layer_mappings.c4_state
    c4_state_global_layers = layer_mappings.c4_state_global_layers
    num_host_pages, swa_num_host_pages = _deepseek_v4_num_host_pages(
        params=params,
        kvcache=kvcache,
        page_size=page_size,
        swa_page_size=kvcache.swa_page_size,
    )

    logical_host_pool = LogicalHostPool(
        num_host_pages * page_size, page_size, layout=get_memory().hicache_mem_layout
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=logical_host_pool,
            device_pool=kvcache,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
    ]

    if not is_unified_kv:
        swa_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.SWA),
            device_buffers=[
                *kvcache.swa_kv_pool.kv_buffer,
                *mtp_swa_device_buffers,
            ],
            item_bytes=kvcache.swa_kv_pool.bytes_per_page_padded,
            num_host_pages=swa_num_host_pages,
            slot_page_size=kvcache.swa_page_size,
            layout=get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
        )
        swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator
        entries.append(
            build_pool_entry(
                name=PoolName.SWA,
                host_pool=swa_host_pool,
                device_pool=kvcache.swa_kv_pool,
                layer_mapping=swa_layer_mapping,
                transfer_layer_num=transfer_layer_num + len(mtp_swa_device_buffers),
                host_evict_fn=host_swa_evict_fn,
                device_evict_fn=device_swa_evict_fn,
                device_alloc_fn=swa_attn_allocator.alloc,
                device_free_fn=swa_attn_allocator.free,
                packed_draft_device_pools=tuple(mtp_swa_device_buffers),
            )
        )

    if c4_layer_mapping:
        c4_device_buffers, c4_item_bytes = _dsv4_compressed_region_buffers(kvcache, 4)
        c4_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4),
            device_buffers=c4_device_buffers,
            item_bytes=c4_item_bytes,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout=get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
        )
        entries.append(
            build_pool_entry(
                name=PoolName.DEEPSEEK_V4_C4,
                host_pool=c4_host_pool,
                device_pool=kvcache.c4_kv_pool,
                layer_mapping=c4_layer_mapping,
                transfer_layer_num=transfer_layer_num,
            )
        )
        for region in _dsv4_indexer_regions(kvcache, page_size):
            entries.append(
                build_pool_entry(
                    name=region.name,
                    host_pool=DeepSeekV4PagedHostPool(
                        pool_name=str(region.name),
                        device_buffers=region.device_buffers,
                        item_bytes=region.item_bytes,
                        num_host_pages=num_host_pages,
                        slot_page_size=page_size,
                        layout=get_memory().hicache_mem_layout,
                        allocator_type=_get_allocator_type(),
                        page_aligned_only=region.page_aligned_only,
                    ),
                    device_pool=kvcache.c4_indexer_kv_pool,
                    layer_mapping=c4_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                )
            )

        if not is_unified_kv:
            c4_state_host_pool = DeepSeekV4StateHostPool(
                pool_name=str(PoolName.DEEPSEEK_V4_C4_STATE),
                state_pools=[
                    kvcache.compress_state_pools[layer_id]
                    for layer_id in c4_state_global_layers
                ],
                num_host_pages=swa_num_host_pages,
                swa_page_size=kvcache.swa_page_size,
                layout=get_memory().hicache_mem_layout,
                allocator_type=_get_allocator_type(),
            )
            c4_indexer_state_host_pool = DeepSeekV4StateHostPool(
                pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER_STATE),
                state_pools=[
                    kvcache.indexer_compress_state_pools[layer_id]
                    for layer_id in c4_state_global_layers
                ],
                num_host_pages=swa_num_host_pages,
                swa_page_size=kvcache.swa_page_size,
                layout=get_memory().hicache_mem_layout,
                allocator_type=_get_allocator_type(),
            )
            entries.extend(
                [
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
        c128_device_buffers, c128_item_bytes = _dsv4_compressed_region_buffers(
            kvcache, 128
        )
        c128_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C128),
            device_buffers=c128_device_buffers,
            item_bytes=c128_item_bytes,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout=get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
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
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def build_hybrid_mamba_stack(
    *,
    params: CacheInitParams,
    kv_pool: Any,
    mamba_pool: Any,
    full_layer_mapping: dict[int, int],
    mamba_layer_mapping: dict[int, int],
    load_cache_event,
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
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    mtp_draft_device_pools = tuple(
        pool.full_kv_pool if isinstance(pool, HybridLinearKVPool) else pool
        for pool in params.mtp_draft_device_pools
    )
    kv_host_size, mamba_host_size = None, 0
    if get_memory().hicache_size > 0:
        kv_host_size, mamba_host_size = _split_hicache_size(
            get_memory().hicache_size, (kv_pool, mamba_pool)
        )
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=params.page_size,
        use_mla=use_mla,
        host_size=kv_host_size,
        mtp_draft_device_pools=mtp_draft_device_pools,
    )
    if mtp_draft_device_pools:
        full_layer_mapping = _with_mtp_layer_mapping(
            full_layer_mapping,
            transfer_layer_start=transfer_layer_num,
            target_device_layer_num=kv_pool.layer_num,
            draft_layer_num=len(mtp_draft_device_pools),
        )
    mamba_host_pool = MambaPoolHost(
        mamba_pool,
        get_memory().hicache_ratio,
        mamba_host_size,
        allocator_type=_get_allocator_type(),
        layout=get_memory().hicache_mem_layout,
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num + len(mtp_draft_device_pools),
            is_anchor=True,
            packed_draft_device_pools=mtp_draft_device_pools,
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
        params.page_size,
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def build_hybrid_mamba_swa_stack(
    *,
    params: CacheInitParams,
    full_kv_pool: Any,
    swa_kv_pool: Any,
    mamba_pool: Any,
    full_layer_mapping: dict[int, int],
    swa_layer_mapping: dict[int, int],
    mamba_layer_mapping: dict[int, int],
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    host_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    device_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(
        full_layer_mapping | swa_layer_mapping | mamba_layer_mapping
    )
    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator
    mamba_allocator = params.req_to_token_pool.mamba_allocator
    kv_host_size, swa_host_size, mamba_host_size = None, None, 0
    if get_memory().hicache_size > 0:
        kv_host_size, swa_host_size, mamba_host_size = _split_hicache_size(
            get_memory().hicache_size, (full_kv_pool, swa_kv_pool, mamba_pool)
        )
    kv_host_pool = build_kv_host_pool(
        kv_pool=full_kv_pool,
        page_size=page_size,
        use_mla=False,
        host_size=kv_host_size,
        pool_label="full",
    )
    swa_host_pool = build_kv_host_pool(
        kv_pool=swa_kv_pool,
        page_size=page_size,
        use_mla=False,
        host_size=swa_host_size,
        pool_label="swa",
    )
    mamba_host_pool = MambaPoolHost(
        mamba_pool,
        get_memory().hicache_ratio,
        mamba_host_size,
        allocator_type=get_memory().hicache_storage_backend,
        layout=get_memory().hicache_mem_layout,
    )
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
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def build_anchor_sidecar_stack(
    *,
    params: CacheInitParams,
    kv_pool: Any,
    sidecar_pool_name: PoolName,
    full_layer_mapping: dict[int, int],
    load_cache_event,
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
    mtp_draft_device_pools = tuple(
        pool for pool in params.mtp_draft_device_pools if pool.index_k_with_scale_buffer
    )
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=params.page_size,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
        mtp_draft_device_pools=mtp_draft_device_pools,
    )
    sidecar_host_pool = sidecar_host_pool_factory(kv_host_pool)
    # Expose packed MTP tail layers to the controller's flat transfer builder.
    if mtp_draft_device_pools:
        full_layer_mapping = _with_mtp_layer_mapping(
            full_layer_mapping,
            transfer_layer_start=transfer_layer_num,
            target_device_layer_num=kv_pool.layer_num,
            draft_layer_num=len(mtp_draft_device_pools),
        )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num + len(mtp_draft_device_pools),
            is_anchor=True,
            packed_draft_device_pools=mtp_draft_device_pools,
        ),
        build_pool_entry(
            name=sidecar_pool_name,
            host_pool=sidecar_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num + len(mtp_draft_device_pools),
            packed_draft_device_pools=mtp_draft_device_pools,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        params.page_size,
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def _build_mha_mla_host_pool(
    *,
    pool: Any,
    host_to_device_ratio: float,
    page_size: int,
    layout: str,
    allocator_type: str,
    pool_label: str,
):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    kwargs = dict(
        host_to_device_ratio=host_to_device_ratio,
        host_size=0,
        page_size=page_size,
        layout=layout,
        allocator_type=allocator_type,
        pool_label=pool_label,
    )
    if isinstance(pool, MHATokenToKVPool):
        return get_mha_host_pool_cls(pool)(pool, **kwargs)
    return MLATokenToKVPoolHost(
        pool,
        override_kv_cache_dim=pool.kv_cache_dim,
        **kwargs,
    )


def build_full_draft_pools(
    *,
    draft_kv_pool: Any,
    tree_cache: Any,
) -> tuple[list[SidecarPoolSpec], list[PoolEntry]]:
    """Build draft KV/DSA sidecars whose indices follow target full KV."""
    from sglang.srt.mem_cache.memory_pool import (
        DSATokenToKVPool,
        HybridLinearKVPool,
    )

    pool = draft_kv_pool
    if isinstance(pool, HybridLinearKVPool):
        # Hybrid draft runners keep their sole attention layer in this sub-pool.
        pool = pool.full_kv_pool
    if pool.layer_num == 0:
        return [], []

    controller = tree_cache.cache_controller
    host_pool_group = controller.mem_pool_host

    # Note(kpham-sgl): DCP x DSpark draft KV is replicated and spans the virtual
    # loc space, so match the target host's logical_size instead of physical size.
    draft_host_pool = _build_mha_mla_host_pool(
        pool=pool,
        host_to_device_ratio=host_pool_group.logical_size / pool.size,
        page_size=controller.page_size,
        layout=get_memory().hicache_mem_layout,
        allocator_type=_get_allocator_type(),
        pool_label="draft",
    )
    draft_layer_mapping = {i: i for i in range(pool.layer_num)}

    specs = [
        SidecarPoolSpec(
            pool_name=PoolName.DRAFT,
            indices_from_pool=PoolName.KV,
        )
    ]
    entries = [
        build_pool_entry(
            name=PoolName.DRAFT,
            host_pool=draft_host_pool,
            device_pool=pool,
            layer_mapping=draft_layer_mapping,
            transfer_layer_num=draft_host_pool.layer_num,
        )
    ]

    if isinstance(pool, DSATokenToKVPool) and pool.index_k_with_scale_buffer:
        indexer_host_pool = DSAIndexerPoolHost(
            pool,
            draft_host_pool,
            get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
        )
        specs.append(
            SidecarPoolSpec(
                pool_name=PoolName.DRAFT_INDEXER,
                indices_from_pool=PoolName.KV,
            )
        )
        entries.append(
            build_pool_entry(
                name=PoolName.DRAFT_INDEXER,
                host_pool=indexer_host_pool,
                device_pool=pool,
                layer_mapping=draft_layer_mapping,
                transfer_layer_num=indexer_host_pool.layer_num,
            )
        )

    return specs, entries


def build_swa_draft_pools(
    *,
    draft_kv_pool: Any,
    tree_cache: Any,
) -> tuple[list[SidecarPoolSpec], list[PoolEntry]]:
    """Build a draft SWA sidecar whose indices follow target SWA."""
    draft_swa_pool = draft_kv_pool.swa_kv_pool
    if draft_swa_pool is None:
        raise NotImplementedError(
            "HiCache draft SWA sidecar requires a non-unified draft SWA pool."
        )
    if draft_swa_pool.layer_num == 0:
        return [], []
    controller = tree_cache.cache_controller
    host_pool_group = controller.mem_pool_host
    target_swa_host_pool = host_pool_group.entry_map[PoolName.SWA].host_pool

    if isinstance(target_swa_host_pool, DeepSeekV4PagedHostPool):
        host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DRAFT_SWA),
            device_buffers=draft_swa_pool.kv_buffer,
            item_bytes=draft_swa_pool.bytes_per_page_padded,
            num_host_pages=target_swa_host_pool.num_host_pages,
            slot_page_size=draft_swa_pool.page_size,
            layout=target_swa_host_pool.layout,
            allocator_type=_get_allocator_type(),
        )
    else:
        host_pool = _build_mha_mla_host_pool(
            pool=draft_swa_pool,
            host_to_device_ratio=target_swa_host_pool.size / draft_swa_pool.size,
            page_size=target_swa_host_pool.page_size,
            layout=target_swa_host_pool.layout,
            allocator_type=_get_allocator_type(),
            pool_label="draft_swa",
        )

    layer_mapping = {i: i for i in range(draft_swa_pool.layer_num)}
    spec = SidecarPoolSpec(
        pool_name=PoolName.DRAFT_SWA,
        indices_from_pool=PoolName.SWA,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    entry = build_pool_entry(
        name=PoolName.DRAFT_SWA,
        host_pool=host_pool,
        device_pool=draft_swa_pool,
        layer_mapping=layer_mapping,
        transfer_layer_num=host_pool.layer_num,
    )
    return [spec], [entry]


def build_hicache_draft_sidecars(
    *,
    draft_device_pools: tuple[Any, ...],
    tree_cache: Any,
) -> tuple[list[SidecarPoolSpec], list[PoolEntry]]:
    """Compose the full and SWA draft-sidecar paths."""
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

    assert len(draft_device_pools) == 1
    draft_kv_pool = draft_device_pools[0]
    builder = (
        build_swa_draft_pools
        if isinstance(draft_kv_pool, BaseSWAKVPool)
        else build_full_draft_pools
    )
    return builder(
        draft_kv_pool=draft_kv_pool,
        tree_cache=tree_cache,
    )


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

    def build_direct_linker_pool_group(
        self,
        *,
        kvcache: Any,
        params: CacheInitParams,
        page_size: int,
    ):
        raise ValueError(
            "The selected hybrid pool strategy does not support the direct "
            f"external linker: {type(self).__name__}."
        )

    def build(
        self,
        *,
        cache: UnifiedRadixCache,
        kvcache: Any,
        params: CacheInitParams,
        server_args: ServerArgs,
        load_cache_event,
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

    def build_direct_linker_pool_group(self, *, kvcache, params, page_size):
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            _build_deepseek_v4_device_pool_group,
        )

        return _build_deepseek_v4_device_pool_group(kvcache, page_size)

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        layer_mappings = _resolve_deepseek_v4_layer_mappings(kvcache)
        host_pool_group, cache_controller = build_deepseek_v4_hicache_stack(
            params=params,
            kvcache=kvcache,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
            device_swa_evict_fn=lambda n: _evict_swa_for_device_alloc(cache, n),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
            layer_mappings=layer_mappings,
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
                (PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C128, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C4_STATE, PoolName.SWA),
                (PoolName.DEEPSEEK_V4_C4_INDEXER_STATE, PoolName.SWA),
                (PoolName.DEEPSEEK_V4_C128_STATE, PoolName.SWA),
            )
            if name in host_pool_group.entry_map
        ]
        component_host_pools = {
            ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
        }
        if PoolName.SWA in host_pool_group.entry_map:
            component_host_pools[ComponentType.SWA] = host_pool_group.get_pool(
                PoolName.SWA
            )

        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools=component_host_pools,
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
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        full_layer_mapping = dict(kvcache.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
        host_pool_group, cache_controller = build_hybrid_mamba_stack(
            params=params,
            kv_pool=kvcache.full_kv_pool,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping=full_layer_mapping,
            mamba_layer_mapping=mamba_layer_mapping,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            use_mla=kvcache.use_mla,
            host_mamba_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA),
            device_mamba_evict_fn=lambda n: _evict_mamba_for_device_alloc(cache, n),
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
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        full_layer_mapping, swa_layer_mapping = _swa_layer_mappings(kvcache)
        host_pool_group, cache_controller = build_hybrid_swa_stack(
            params=params,
            full_kv_pool=kvcache.full_kv_pool,
            swa_kv_pool=kvcache.swa_kv_pool,
            full_layer_mapping=full_layer_mapping,
            swa_layer_mapping=swa_layer_mapping,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            use_mla=False,
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
            device_swa_evict_fn=lambda n: _evict_swa_for_device_alloc(cache, n),
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
            pools_desc="Full + SWA",
        )


class _MambaSwaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        return (
            isinstance(kvcache, SWAKVPool)
            and not isinstance(kvcache, DeepSeekV4TokenToKVPool)
            and components
            == {ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA}
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
        full_layer_mapping, swa_layer_mapping = _swa_layer_mappings(kvcache)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
        host_pool_group, cache_controller = build_hybrid_mamba_swa_stack(
            params=params,
            full_kv_pool=kvcache.full_kv_pool,
            swa_kv_pool=kvcache.swa_kv_pool,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping=full_layer_mapping,
            swa_layer_mapping=swa_layer_mapping,
            mamba_layer_mapping=mamba_layer_mapping,
            page_size=cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=params.pp_cache_group,
            storage_backend=storage_backend,
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
            device_swa_evict_fn=lambda n: _evict_swa_for_device_alloc(cache, n),
            host_mamba_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA),
            device_mamba_evict_fn=lambda n: _evict_mamba_for_device_alloc(cache, n),
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
                ComponentType.MAMBA: host_pool_group.get_pool(PoolName.MAMBA),
            },
            register_req_to_token_counter=True,
            transfer_layer_num=len(
                full_layer_mapping | swa_layer_mapping | mamba_layer_mapping
            ),
            pools_desc="KV + SWA + MAMBA",
        )


class _DsaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        return isinstance(kvcache, DSATokenToKVPool) and components == {
            ComponentType.FULL
        }

    def build_direct_linker_pool_group(self, *, kvcache, params, page_size):
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            _build_dsa_device_pool_group,
        )

        return _build_dsa_device_pool_group(kvcache, page_size)

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
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
            kv_pool=full_kv_pool,
            sidecar_pool_name=PoolName.INDEXER,
            full_layer_mapping=full_layer_mapping,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            use_mla=use_mla,
            override_kv_cache_dim=full_kv_pool.kv_cache_dim,
            sidecar_host_pool_factory=lambda kv_host_pool: DSAIndexerPoolHost(
                full_kv_pool,
                kv_host_pool,
                get_memory().hicache_mem_layout,
                allocator_type=_get_allocator_type(),
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


class _MiniMaxSparseStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

        return isinstance(kvcache, MiniMaxSparseKVPool) and components == {
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
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        host_pool_group, cache_controller = build_minimax_sparse_hicache_stack(
            params=params,
            sparse_pool=kvcache,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        sidecars = []
        pools_desc = "KV"
        if kvcache.index_k_pool is not None:
            sidecars.append(
                SidecarPoolSpec(
                    pool_name=PoolName.INDEXER,
                    indices_from_pool=PoolName.KV,
                )
            )
            pools_desc = "KV + INDEXER(k-only)"
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
            },
            sidecars=sidecars,
            transfer_layer_num=kvcache.main_pool.layer_num,
            pools_desc=pools_desc,
        )


class _PlainKvStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.memory_pool import (
            DSATokenToKVPool,
            HybridLinearKVPool,
            MiniMaxSparseKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        if isinstance(
            kvcache,
            (
                SWAKVPool,
                HybridLinearKVPool,
                DSATokenToKVPool,
                MiniMaxSparseKVPool,
                DeepSeekV4TokenToKVPool,
            ),
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
            kv_pool=full_kv_pool,
            full_layer_mapping=full_layer_mapping,
            load_cache_event=load_cache_event,
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
    _MambaSwaStrategy(),
    _DsaStrategy(),
    _MiniMaxSparseStrategy(),
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
            storage_backend=storage_backend,
            storage_backend_extra_config=storage_extra_config,
            prefetch_threshold=storage_prefetch_threshold,
            model_name=get_serving().served_model_name,
            enable_storage_metrics=cache._enable_metrics_flag,
        )
        _apply_stack_result(cache, kvcache, params, result)
    except Exception:
        logger.exception("attach_hybrid_pool_to_unified_cache failed")
        raise


def build_minimax_sparse_hicache_stack(
    *,
    params: CacheInitParams,
    sparse_pool: Any,
    load_cache_event,
    storage_backend: Optional[str],
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    """KV (main_pool) + INDEXER (index_k_pool) host stack for MiniMax M3 sparse."""
    # Mappings are stage-local keyed (controller iterates 0..transfer_layer_num).
    # PP>1 stays gated below pending end-to-end validation of the sparse host path.
    if params.pp_size > 1:
        raise NotImplementedError(
            "MiniMax-M3 sparse HiCache does not support pipeline parallelism "
            "(pp_size>1) yet."
        )
    # mirror HiRadix's guard, which the Unified-tree strategy path otherwise skips.
    if sparse_pool.index_kv_pool is not None:
        raise ValueError(
            "MiniMax sparse HiCache currently supports index-k-only sparse layers; "
            "index_kv_pool (value-bearing) layers are not cached/restored yet."
        )
    main_pool = sparse_pool.main_pool
    start_layer = main_pool.start_layer
    transfer_layer_num = main_pool.layer_num
    # Stage-local keys (0..transfer_layer_num) match the controller's per-layer
    # load loop; values index the host pool's local layer buffer.
    full_layer_mapping = {layer_id: layer_id for layer_id in range(transfer_layer_num)}

    kv_host_pool = build_kv_host_pool(
        kv_pool=main_pool,
        page_size=params.page_size,
        use_mla=False,
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=main_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
    ]

    index_k_pool = sparse_pool.index_k_pool
    if index_k_pool is not None:
        index_host_pool = MHATokenToKOnlyPoolHost(
            index_k_pool,
            kv_host_pool,
            get_memory().hicache_mem_layout,
            allocator_type=get_memory().hicache_storage_backend,
        )
        entries.append(
            build_pool_entry(
                name=PoolName.INDEXER,
                host_pool=index_host_pool,
                device_pool=index_k_pool,
                layer_mapping={
                    gid - start_layer: sub_id
                    for gid, sub_id in sparse_pool.index_k_layer_id_mapping.items()
                },
                transfer_layer_num=transfer_layer_num,
            )
        )

    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        params.page_size,
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_group=params.pp_cache_group,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def attach_hybrid_minimax_sparse_pool_to_hiradix_cache(
    radix_cache: HiRadixCache,
    params: CacheInitParams,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
) -> None:
    """Attach HostPoolGroup (KV + index K) + HybridCacheController for HiRadixCache."""
    from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

    try:
        sparse_pool = radix_cache.kv_cache
        if not isinstance(sparse_pool, MiniMaxSparseKVPool):
            raise TypeError(
                f"Expected MiniMaxSparseKVPool, got {type(sparse_pool).__name__}"
            )
        if sparse_pool.index_kv_pool is not None:
            raise ValueError(
                "MiniMax M3 HiCache L2 currently supports index-k-only sparse layers "
                "(sparse_disable_index_value=1 for all sparse layers). "
                "This model has index_kv_pool layers; INDEXER_KV sidecar is not "
                "implemented yet."
            )

        main_pool = sparse_pool.main_pool
        if sparse_pool.index_k_pool is None:
            host_pool_group, cache_controller = build_kv_only_stack(
                params=params,
                kv_pool=main_pool,
                full_layer_mapping={
                    layer_id: layer_id for layer_id in range(main_pool.layer_num)
                },
                load_cache_event=load_cache_event,
                storage_backend=get_memory().hicache_storage_backend,
                use_mla=False,
                prefetch_threshold=prefetch_threshold,
                model_name=get_serving().served_model_name,
                storage_backend_extra_config=extra_config,
                enable_storage_metrics=enable_storage_metrics,
            )
            pools_desc = "KV"
        else:
            host_pool_group, cache_controller = build_minimax_sparse_hicache_stack(
                params=params,
                sparse_pool=sparse_pool,
                load_cache_event=load_cache_event,
                storage_backend=get_memory().hicache_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=get_serving().served_model_name,
                storage_backend_extra_config=extra_config,
                enable_storage_metrics=enable_storage_metrics,
            )
            pools_desc = "KV + INDEXER(k-only)"

        sparse_pool.register_layer_transfer_counter(cache_controller.layer_done_counter)
        radix_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
        radix_cache.token_to_kv_pool_host = host_pool_group
        radix_cache.cache_controller = cache_controller
        logger.info(
            "Attached hybrid MiniMax sparse pool stack to HiRadixCache: pools=%s, "
            "transfer_layer_num=%s, sparse_index_k_layers=%s",
            pools_desc,
            main_pool.layer_num,
            len(sparse_pool.index_k_layer_id_mapping),
        )
    except Exception:
        logger.exception("attach_hybrid_minimax_sparse_pool_to_hiradix_cache failed")
        raise


def attach_hybrid_dsa_pool_to_hiradix_cache(
    radix_cache: HiRadixCache,
    params: CacheInitParams,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
) -> None:
    """Attach HostPoolGroup (KV + indexer) + HybridCacheController for HiRadixCache.

    This entrypoint is currently intended only for HiRadixCache's DSA path.
    """
    try:
        kv = radix_cache.kv_cache
        layer_mapping = {layer_id: layer_id for layer_id in range(kv.layer_num)}
        host_pool_group, cache_controller = build_anchor_sidecar_stack(
            params=params,
            kv_pool=kv,
            sidecar_pool_name=PoolName.INDEXER,
            full_layer_mapping=layer_mapping,
            load_cache_event=load_cache_event,
            storage_backend=get_memory().hicache_storage_backend,
            use_mla=True,
            override_kv_cache_dim=kv.kv_cache_dim,
            prefetch_threshold=prefetch_threshold,
            sidecar_host_pool_factory=lambda kv_host_pool: DSAIndexerPoolHost(
                kv,
                kv_host_pool,
                get_memory().hicache_mem_layout,
                allocator_type=_get_allocator_type(),
            ),
            model_name=get_serving().served_model_name,
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
