"""Capacity questions the scheduler asks of a prefix cache and its allocator.

Each answer may need eviction first: the prefix cache chooses what to evict
and the allocator says whether the demand fits afterwards. A pool whose
sub-pools share one memory (`pools_share_memory`) answers by counts and
geometry alone and never calls the cache; this module drives "evict, then
check" for it. Separate pools keep their own hooks, which reclaim from the
cache they are handed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.kv.alloc.demand import KVDemand

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


def _shares_prefixes(tree_cache: Optional[BasePrefixCache]) -> bool:
    return tree_cache is not None and tree_cache.supports_prefix_sharing()


def _evict(tree_cache: BasePrefixCache, allocator, demand: KVDemand) -> None:
    # Frees a previous step queued go back first, so the cache's walk sees
    # the room they make and does not evict for it.
    allocator.flush_deferred_full_frees()
    tree_cache.evict_for_alloc(demand)


def reclaim_for_prealloc(
    tree_cache: Optional[BasePrefixCache], allocator, demand: KVDemand
) -> Optional[str]:
    """Free room for a preallocation of ``demand``; None means it is ready,
    else a description of the shortfall for the caller to attach to its
    request."""
    if not allocator.pools_share_memory:
        return allocator.reclaim_for_prealloc(tree_cache, demand)
    if _shares_prefixes(tree_cache):
        _evict(tree_cache, allocator, demand)
    return allocator.prealloc_shortfall(demand)


def admit_preallocation(
    tree_cache: Optional[BasePrefixCache],
    allocator,
    admission: KVDemand,
    take: KVDemand,
    *,
    evictable,
    full_budget_tokens: int,
    swa_budget_tokens: Optional[int] = None,
) -> bool:
    """Whether a decode-node preallocation is admitted (``admission``, with its
    decode reservation, crediting what the cache could evict) and its rows
    (``take``) can be taken now: every admission that credits eviction
    reclaims before the take."""
    if not allocator.pools_share_memory:
        return allocator.admit_preallocation(
            tree_cache,
            admission,
            take,
            evictable=evictable,
            full_budget_tokens=full_budget_tokens,
            swa_budget_tokens=swa_budget_tokens,
        )
    if not allocator.prealloc_fits(
        admission,
        evictable=evictable,
        full_budget_tokens=full_budget_tokens,
        swa_budget_tokens=swa_budget_tokens,
    ):
        return False
    return reclaim_for_prealloc(tree_cache, allocator, take) is None


def reclaim_for_load_back(
    tree_cache: Optional[BasePrefixCache], allocator, demand: KVDemand
) -> Optional[str]:
    """Reclaim before a host load takes its rows; None means ready."""
    if not allocator.pools_share_memory:
        return allocator.reclaim_for_load_back(tree_cache, demand)
    return reclaim_for_prealloc(tree_cache, allocator, demand)


def evict_to_free_tokens(
    tree_cache: Optional[BasePrefixCache], allocator, num_tokens: int
) -> Optional[bool]:
    """Evict cached entries until ``num_tokens`` new tokens, which every
    window keeps too, can be taken. Return whether they can, or None when
    there is no prefix cache to evict from (or the pool does not say)."""
    if not allocator.pools_share_memory:
        return allocator.evict_to_free_tokens(tree_cache, num_tokens)
    if not _shares_prefixes(tree_cache):
        return None
    demand = KVDemand.same_tokens(allocator, num_tokens)
    _evict(tree_cache, allocator, demand)
    return allocator.units_fit(
        {n: d for n, d in allocator.page_demands(demand).items() if d > 0}
    )


def check_decode_capacity(
    tree_cache: Optional[BasePrefixCache],
    allocator,
    *,
    num_tokens: int,
    requests=None,
    spec_algorithm=None,
) -> bool:
    """Whether the next decode step's ``num_tokens`` allocation fits after
    evicting reclaimable cache: the same question the decode take answers."""
    if not allocator.pools_share_memory:
        return allocator.check_decode_capacity(
            num_tokens=num_tokens,
            tree_cache=tree_cache,
            requests=requests,
            spec_algorithm=spec_algorithm,
        )
    evict_to_free_tokens(tree_cache, allocator, num_tokens)
    return allocator.ensure_capacity(KVDemand.same_tokens(allocator, num_tokens))


def create_prefill_budget(
    allocator, tree_cache: Optional[BasePrefixCache], *, num_mixed_decode_tokens=0
):
    """The prefill admission budget over ``allocator`` and ``tree_cache``."""
    if not allocator.pools_share_memory:
        return allocator.create_prefill_budget(
            tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )
    from sglang.srt.mem_cache.prefill_budget import ChainPrefillBudget

    return ChainPrefillBudget(
        allocator, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
    )
