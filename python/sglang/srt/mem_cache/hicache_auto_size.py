from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.pool_host.base import (
    host_memory_budget_bytes,
    host_memory_budget_scope,
    ranks_per_host,
)
from sglang.srt.runtime_context import get_context, get_memory, get_parallel

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.speculative.base_spec_worker import HiCacheDraftPlan

logger = logging.getLogger(__name__)

# Page rounding, allocator metadata and staging buffers are outside the device
# byte counts the ratio is derived from.
_ALLOCATION_SLACK_FRACTION = 0.05

# Leaf pools; a pool wrapping sub-pool views is sized by its views.
_SIZEABLE_POOLS = (
    MHATokenToKVPool,
    MLATokenToKVPool,
    MiniMaxSparseKVPool,
)


def _sub_pool_views(pool) -> list:
    """The token KV views a pool wraps; empty for a pool that is its own view."""
    router = pool.layer_router
    views = [router.view(sub_pool.name) for sub_pool in router.sub_pools()]
    return [] if len(views) == 1 and views[0] is pool else views


def _pool_bytes(pool) -> int:
    views = _sub_pool_views(pool)
    if views:
        return sum(_pool_bytes(view) for view in views)
    return _leaf_bytes(pool)


def _leaf_bytes(pool) -> int:
    sizes = getattr(pool, "host_capacity_bytes", None)
    if sizes is None:
        sizes = pool.get_kv_size_bytes()
    return sum(sizes) if isinstance(sizes, tuple) else sizes


def _draft_bytes(target, draft) -> int:
    if draft.layer_router.sub_pool(
        IdSpaceKind.SLIDING_WINDOW
    ) is not None or isinstance(draft, DeepSeekV4TokenToKVPool):
        # Match sidecar construction: only SWA drafts follow target SWA slots.
        target = (
            target.layer_router.view(IdSpaceKind.SLIDING_WINDOW)
            if target.layer_router.sub_pool(IdSpaceKind.SLIDING_WINDOW) is not None
            else target.swa_kv_pool
        )
        draft = draft.swa_kv_pool
    # A sidecar has one host slot per target slot, however few slots the draft has.
    return _pool_bytes(draft) * target.size // draft.size


def _estimate_hicache_bytes(
    params: CacheInitParams, draft_plan: HiCacheDraftPlan | None
) -> int:
    """Device bytes whose host mirrors scale with the HiCache ratio."""
    # Every token KV view the pool routes to, and the recurrent state beside it.
    pool = params.token_to_kv_pool_allocator.get_kvcache()
    if not (isinstance(pool, _SIZEABLE_POOLS) or _sub_pool_views(pool)):
        raise ValueError(
            f"HiCache auto-sizing does not support {type(pool).__name__}; "
            "set --hicache-ratio or --hicache-size explicitly."
        )
    total = _pool_bytes(pool)
    if params.req_to_token_pool.holds_state_slots:
        total += _leaf_bytes(params.token_to_kv_pool_allocator.state.pool)
    drafts = params.mtp_draft_device_pools
    if draft_plan is not None and draft_plan.mode == "sidecar":
        drafts = draft_plan.device_pools
    return total + sum(_draft_bytes(pool, draft) for draft in drafts)


@contextmanager
def auto_size_hicache(
    params: CacheInitParams, draft_plan: HiCacheDraftPlan | None, *, enabled: bool
):
    """Reduce the default HiCache ratio until this machine's host pools fit.

    Resolution nulls the fraction for an explicit --hicache-ratio/--hicache-size.
    """
    fraction = get_memory().hicache_host_memory_fraction
    if not enabled or fraction is None:
        yield
        return
    requested = get_memory().hicache_ratio
    device_bytes = _estimate_hicache_bytes(params, draft_plan)
    budget = int(host_memory_budget_bytes(auto_size=True) * fraction)
    ratio = min(requested, budget * (1 - _ALLOCATION_SLACK_FRACTION) / device_bytes)
    # One collective before any pool is built: PP stages own different pool
    # counts, so a per-pool collective could deadlock.
    if torch.distributed.is_initialized():
        value = torch.tensor([ratio], dtype=torch.float64)
        torch.distributed.all_reduce(
            value,
            op=torch.distributed.ReduceOp.MIN,
            group=get_parallel().world_group.cpu_group,
        )
        ratio = value.item()
    if ratio <= 0:
        raise ValueError(
            "No host memory is left for HiCache after the 10 GiB reserve; "
            "set --hicache-ratio or --hicache-size explicitly."
        )
    get_context().override("hicache.auto_size", hicache_ratio=ratio)
    logger.info(
        "HiCache auto-sizing: ratio %.3f -> %.3f; %.1f GiB host memory per rank "
        "(fraction %.2f, %d ranks on this host), host pools %.1f GiB.",
        requested,
        ratio,
        budget / 1024**3,
        fraction,
        ranks_per_host(),
        device_bytes * ratio / 1024**3,
    )
    with host_memory_budget_scope(budget):
        yield
