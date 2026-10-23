from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.mla_host_dedup import (
    mla_dedup_rank_and_size,
    mla_host_dedup_eligible,
)
from sglang.srt.mem_cache.pool_host.base import (
    host_memory_budget_bytes,
    host_memory_budget_scope,
    ranks_per_host,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_context, get_memory, get_parallel

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.speculative.base_spec_worker import HiCacheDraftPlan

logger = logging.getLogger(__name__)

# Page rounding, allocator metadata and staging buffers are outside the device
# byte counts the ratio is derived from.
_ALLOCATION_SLACK_FRACTION = 0.05

_SIZEABLE_POOLS = (
    MHATokenToKVPool,
    MLATokenToKVPool,
    SWAKVPool,
    HybridLinearKVPool,
    MiniMaxSparseKVPool,
)


def _pool_bytes(pool) -> int:
    if isinstance(pool, SWAKVPool):
        return _pool_bytes(pool.full_kv_pool) + _pool_bytes(pool.swa_kv_pool)
    if isinstance(pool, HybridLinearKVPool):
        return _pool_bytes(pool.full_kv_pool)
    sizes = getattr(pool, "host_capacity_bytes", None)
    if sizes is None:
        sizes = pool.get_kv_size_bytes()
    return sum(sizes) if isinstance(sizes, tuple) else sizes


def _draft_bytes(target, draft) -> int:
    if isinstance(draft, BaseSWAKVPool):
        # Match sidecar construction: only SWA drafts follow target SWA slots.
        target, draft = target.swa_kv_pool, draft.swa_kv_pool
    # A sidecar has one host slot per target slot, however few slots the draft has.
    return _pool_bytes(draft) * target.size // draft.size


def _estimate_hicache_bytes(
    params: CacheInitParams,
    draft_plan: HiCacheDraftPlan | None,
    *,
    physical: bool = False,
) -> int:
    """Device bytes whose host mirrors scale with the HiCache ratio."""
    pool = params.token_to_kv_pool_allocator.get_kvcache()
    if not isinstance(pool, _SIZEABLE_POOLS):
        raise ValueError(
            f"HiCache auto-sizing does not support {type(pool).__name__}; "
            "set --hicache-ratio or --hicache-size explicitly."
        )
    total = _pool_bytes(pool)
    target_bytes = total
    if physical and get_memory().enable_mla_hicache_host_dedup:
        target = pool
        rank, size = mla_dedup_rank_and_size()
        if size > 1 and rank != 0 and mla_host_dedup_eligible(target, None):
            # Only the target MLA/DSA mirror is deduplicated. Draft and linear
            # state below remain physical allocations on every rank.
            total -= target_bytes
    if isinstance(params.req_to_token_pool, HybridReqToTokenPool):
        total += _pool_bytes(params.req_to_token_pool.mamba_pool)
    drafts = params.mtp_draft_device_pools
    if draft_plan is not None and draft_plan.mode == "sidecar":
        drafts = draft_plan.device_pools
    return total + sum(_draft_bytes(pool, draft) for draft in drafts)


def _dedup_host_budget(physical_bytes: int, rank_budget: int) -> tuple[float, int]:
    """Share a host's allowance by physical demand before constructing any pool.

    Use the smallest cgroup-bounded snapshot reported on a node. This stays
    conservative when ranks have different visible container allowances.
    """
    # Pool byte counts may be NumPy int64 values. The proportional allocation
    # multiplies two byte counts, which can exceed int64 even when each fits.
    physical_bytes = int(physical_bytes)
    rank_budget = int(rank_budget)
    node_rank = get_parallel().node_rank
    local = torch.tensor(
        [node_rank, rank_budget * ranks_per_host(), physical_bytes], dtype=torch.int64
    )
    records = [local]
    if torch.distributed.is_initialized():
        group = get_parallel().world_group.cpu_group
        records = [
            torch.empty_like(local)
            for _ in range(torch.distributed.get_world_size(group=group))
        ]
        torch.distributed.all_gather(records, local, group=group)
    peers = [record.tolist() for record in records if record[0].item() == node_rank]
    host_budget = min(record[1] for record in peers)
    host_bytes = sum(record[2] for record in peers)
    if host_bytes == 0:
        # A node containing only dummy peers must not constrain an owner on a
        # different node. It still participates in the subsequent global MIN.
        return float("inf"), 0
    return host_budget / host_bytes, host_budget * physical_bytes // host_bytes


@contextmanager
def auto_size_hicache(
    params: CacheInitParams, draft_plan: HiCacheDraftPlan | None, *, enabled: bool
):
    """Reduce the default HiCache ratio until this machine's host pools fit.

    Explicit dedup ratios retain their capacity but share physical budgets.
    Explicit --hicache-size still uses the original per-rank allocation check.
    """
    memory = get_memory()
    fraction = memory.hicache_host_memory_fraction
    dedup = memory.enable_mla_hicache_host_dedup
    automatic = fraction is not None
    if not enabled or (not automatic and (not dedup or memory.hicache_size > 0)):
        yield
        return
    requested = memory.hicache_ratio
    device_bytes = _estimate_hicache_bytes(params, draft_plan, physical=dedup)
    budget = int(
        host_memory_budget_bytes(auto_size=automatic) * (fraction if automatic else 1.0)
    )
    if dedup:
        capacity_ratio, budget = _dedup_host_budget(device_bytes, budget)
    else:
        capacity_ratio = budget / device_bytes
    ratio = (
        min(requested, capacity_ratio * (1 - _ALLOCATION_SLACK_FRACTION))
        if automatic
        else capacity_ratio
    )
    # All collectives run before any pool is built: PP stages own different
    # pool counts, so a per-pool collective could deadlock.
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
    if not automatic:
        if requested > ratio:
            raise ValueError(
                f"Not enough host memory for explicit HiCache ratio {requested}; "
                f"the deduplicated pools fit at most ratio {ratio:.3f} before "
                "page rounding. Reduce --hicache-ratio."
            )
        # Honor the caller's explicit capacity; only redistribute its physical
        # allocation allowance, without silently shrinking the ratio.
        with host_memory_budget_scope(budget):
            yield
        return
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
