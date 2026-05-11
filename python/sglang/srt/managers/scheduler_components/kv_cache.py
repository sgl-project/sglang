from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass  # noqa: F401
from typing import Optional  # noqa: F401

from sglang.srt.configs.model_config import ModelImpl  # noqa: F401
from sglang.srt.environ import envs  # noqa: F401
from sglang.srt.managers.mm_utils import init_mm_embedding_cache  # noqa: F401
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: F401
from sglang.srt.mem_cache.radix_cache import RadixCache  # noqa: F401
from sglang.srt.model_loader.utils import get_resolved_model_impl  # noqa: F401
from sglang.srt.session.streaming_session import StreamingSession  # noqa: F401


@dataclass(frozen=True, slots=True, kw_only=True)
class KVCacheBuildResult:
    """Return type for ``build_kv_cache``: 9 fields the caller writes back to
    ``Scheduler.self.X``. Field-cluster bundling (a single
    ``self._kv_cache`` ref instead of 9) is a follow-up commit."""

    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    sliding_window_size: Optional[int]
    full_tokens_per_layer: Optional[int]
    swa_tokens_per_layer: Optional[int]
    req_to_token_pool: object
    token_to_kv_pool_allocator: object
    disable_radix_cache: bool
    tree_cache: object


def get_draft_kv_pool(
    *,
    draft_worker,
    spec_algorithm,
    server_args,
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
    tree_cache,
    draft_worker,
    spec_algorithm,
    server_args,
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
