"""Assemble the optional compressed Host pool with the existing controller."""

import logging

from sglang.srt.kv_compression.layout import KVLayoutAdapter
from sglang.srt.kv_compression.runtime import KVCompressionRuntime
from sglang.srt.kv_compression.store import CompressedHostKVCache
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

logger = logging.getLogger(__name__)
import json


def attach_compressed_hicache(cache, params, load_cache_event):
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
    from sglang.srt.mem_cache.pool_host.base import host_memory_budget_bytes
    from sglang.srt.runtime_context import get_memory

    pool = params.token_to_kv_pool_allocator.get_kvcache()
    layout = KVLayoutAdapter(pool.k_buffer, pool.v_buffer, params.page_size)
    budget = int(get_memory().hicache_size * 1e9)
    if budget > host_memory_budget_bytes(budget):
        raise ValueError(
            "Compressed L2 exceeds the available HiCache host memory budget"
        )
    force = envs.SGLANG_PD_KV_COMPRESSION_FORCE.get()
    verify = envs.SGLANG_PD_KV_COMPRESSION_VERIFY.get()
    runtime = KVCompressionRuntime(
        layout,
        envs.SGLANG_HICACHE_KV_COMPRESSION.get(),
        envs.SGLANG_KV_COMPRESSION_WORKSPACE_MB.get() * 1024**2,
        force=force,
        verify=verify,
    )
    try:
        host = CompressedHostKVCache(
            layout.page_bytes,
            budget,
            verify=verify,
            reservation_bytes=max(layout.page_bytes, runtime.output_bound)
            if force
            else None,
        )
    except BaseException:
        runtime.close()
        raise
    logger.info(
        "KV_COMPRESSION_CAPACITY %s",
        json.dumps(
            dict(
                page_bytes=layout.page_bytes,
                output_bound=runtime.output_bound,
                workspace_bytes=runtime.budget_bytes,
                backup_window_pages=runtime.batch_pages,
                backup_window_workspace_bytes=runtime.batch_pages
                * (2 * layout.page_bytes + runtime.output_bound),
                host_arena_bytes=host.arena.numel(),
                reservation_bytes=host.reservation_bytes,
                force=force,
                verify=verify,
                max_reserved_host_pages=host.arena.numel() // host.reservation_bytes,
            )
        ),
    )
    group = HostPoolGroup(
        [PoolEntry(PoolName.KV, host, pool, lambda layer: layer, True)]
    )
    controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        group,
        params.page_size,
        params.tp_cache_group,
        load_cache_event,
        write_policy="write_through",
        io_backend="kernel",
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
    )
    cache.host_pool_group = group
    cache.cache_controller = controller
    cache.full_kv_pool_host = host
    cache.components[ComponentType.FULL]._full_kv_pool_host = host
    from sglang.srt.kv_compression.host_io import CompressedHostIO
    from sglang.srt.kv_compression.provider import (
        HostEncodedKVProvider,
        RepresentationSpec,
    )
    from sglang.srt.mem_cache.l2_completion import AsyncL2State

    spec = RepresentationSpec(layout.tag, runtime.mode, runtime.force, runtime.verify)
    provider = HostEncodedKVProvider(host, spec)
    runtime.provider = provider
    host.io = CompressedHostIO(runtime, host)
    controller.async_l2 = AsyncL2State(runtime, host, provider)
    cache.tree_core.on_full_kv_recomputed = cache.on_kv_recomputed
