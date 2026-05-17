from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:

    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def get_draft_kv_pool(
    *,
    draft_worker: "BaseTpWorker",
    spec_algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
    enable_overlap: bool,
):
    """Return (draft_token_to_kv_pool, draft_model_config) for the current
    draft worker, or (None, None) when no draft KV pool is available."""
    if draft_worker is None or spec_algorithm.is_ngram():
        return None, None

    if spec_algorithm.supports_spec_v2() and enable_overlap:
        if server_args.enable_multi_layer_eagle:
            draft_runner = draft_worker.draft_worker.draft_runner_list[0]
        else:
            draft_runner = draft_worker.draft_worker.draft_runner
        return draft_runner.token_to_kv_pool, draft_runner.model_config

    return (
        draft_worker.model_runner.token_to_kv_pool,
        draft_worker.model_config,
    )


def maybe_register_hicache_draft(
    *,
    tree_cache: "BasePrefixCache",
    draft_worker: "BaseTpWorker",
    spec_algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
    enable_hierarchical_cache: bool,
    enable_overlap: bool,
    page_size: int,
) -> None:
    """Register draft KV pool with HiCacheController for piggyback L2/L3 ops."""
    if not enable_hierarchical_cache:
        return

    draft_kv_pool, _ = get_draft_kv_pool(
        draft_worker=draft_worker,
        spec_algorithm=spec_algorithm,
        server_args=server_args,
        enable_overlap=enable_overlap,
    )
    if draft_kv_pool is None:
        return

    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        MHATokenToKVPool,
        MLATokenToKVPool,
    )
    from sglang.srt.mem_cache.memory_pool_host import (
        MHATokenToKVPoolHost,
        MLATokenToKVPoolHost,
    )

    pool = draft_kv_pool
    if isinstance(pool, HybridLinearKVPool):
        pool = pool.full_kv_pool

    # Create host pool for draft with the same slot count as the target host pool,
    # so that host indices stay 1-to-1 between target and draft KV caches.
    primary = tree_cache.cache_controller.mem_pool_host
    kw = dict(
        host_to_device_ratio=primary.size / pool.size,
        host_size=0,
        page_size=page_size,
        layout=server_args.hicache_mem_layout,
    )
    if isinstance(pool, MHATokenToKVPool):
        draft_host_pool = MHATokenToKVPoolHost(pool, **kw)
    elif isinstance(pool, MLATokenToKVPool):
        draft_host_pool = MLATokenToKVPoolHost(pool, **kw)
    else:
        logger.warning(
            "Draft pool type %s not supported for HiCache, skipping.",
            type(pool).__name__,
        )
        return

    tree_cache.cache_controller.set_draft_kv_pool(pool, draft_host_pool)
