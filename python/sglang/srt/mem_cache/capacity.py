"""Capacity questions the scheduler asks of a prefix cache and its allocator.

The prefix cache chooses what to evict and the allocator says whether the
demand fits afterwards; how the two combine is the pool's reclaim discipline
(`discipline`), chosen from what the allocator declares:

- a pool whose sub-pools share one memory never calls the cache; the cache
  reclaims a demand as one (`_SharedMemory`), window entries first when
  `windows_reclaimed_first`;
- pools of their own keep their own hooks, and the cache evicts each pool's
  shortfall (`_OwnPools`).

Every caller, `evict_for_alloc` included, goes through the discipline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

from sglang.srt.mem_cache.kv.alloc.demand import KVDemand

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictResult


def windows_reclaimed_first() -> bool:
    """Whether reclaim over one shared memory evicts window entries before
    token entries: only with a HiCache storage backend or on a PD decode node,
    where the FULL prefix is worth keeping. Read on every reclaim, since a
    backend can attach and a PD role change while the server runs."""
    from sglang.srt.runtime_context import get_disagg, get_memory

    memory = get_memory()
    return (
        memory.enable_hierarchical_cache and memory.hicache_storage_backend is not None
    ) or get_disagg().disaggregation_mode == "decode"


def _shares_prefixes(tree_cache: Optional[BasePrefixCache]) -> bool:
    return tree_cache is not None and tree_cache.supports_prefix_sharing()


def _evict(tree_cache: BasePrefixCache, allocator, demand: KVDemand) -> None:
    # Frees a previous step queued go back first, so the cache's walk sees
    # the room they make and does not evict for it.
    allocator.flush_deferred_full_frees()
    tree_cache.evict_for_alloc(demand)


class _OwnPools:
    """Pools of their own: the allocator's hooks answer, reclaiming from the
    cache they are handed; the cache evicts each pool's shortfall."""

    def evict_for_alloc(self, cache: BasePrefixCache, demand: KVDemand) -> EvictResult:
        return cache.evict_pool_shortfalls(demand)

    def windows_share_owner_memory(self, allocator) -> bool:
        return False

    def reclaim_for_prealloc(self, tree_cache, allocator, demand):
        return allocator.reclaim_for_prealloc(tree_cache, demand)

    def admit_preallocation(self, tree_cache, allocator, admission, take, **budgets):
        return allocator.admit_preallocation(tree_cache, admission, take, **budgets)

    def reclaim_for_load_back(self, tree_cache, allocator, demand):
        return allocator.reclaim_for_load_back(tree_cache, demand)

    def evict_to_free_tokens(self, tree_cache, allocator, num_tokens):
        return allocator.evict_to_free_tokens(tree_cache, num_tokens)

    def check_decode_capacity(self, tree_cache, allocator, **kwargs):
        return allocator.check_decode_capacity(tree_cache=tree_cache, **kwargs)

    def create_prefill_budget(self, allocator, tree_cache, num_mixed_decode_tokens):
        return allocator.create_prefill_budget(
            tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )


class _SharedMemory:
    """Sub-pools sharing one memory: the cache reclaims a demand as one and
    the allocator answers by counts and geometry."""

    def evict_for_alloc(self, cache: BasePrefixCache, demand: KVDemand) -> EvictResult:
        return cache.reclaim_jointly(demand, windows_first=windows_reclaimed_first())

    def windows_share_owner_memory(self, allocator) -> bool:
        return bool(allocator.window_pool_names())

    def reclaim_for_prealloc(self, tree_cache, allocator, demand):
        if _shares_prefixes(tree_cache):
            _evict(tree_cache, allocator, demand)
        return allocator.prealloc_shortfall(demand)

    def admit_preallocation(self, tree_cache, allocator, admission, take, **budgets):
        if not allocator.prealloc_fits(admission, **budgets):
            return False
        return self.reclaim_for_prealloc(tree_cache, allocator, take) is None

    def reclaim_for_load_back(self, tree_cache, allocator, demand):
        return self.reclaim_for_prealloc(tree_cache, allocator, demand)

    def evict_to_free_tokens(self, tree_cache, allocator, num_tokens):
        if not _shares_prefixes(tree_cache):
            return None
        demand = KVDemand.same_tokens(allocator, num_tokens)
        _evict(tree_cache, allocator, demand)
        return allocator.units_fit(
            {n: d for n, d in allocator.page_demands(demand).items() if d > 0}
        )

    def check_decode_capacity(self, tree_cache, allocator, *, num_tokens, **_):
        self.evict_to_free_tokens(tree_cache, allocator, num_tokens)
        return allocator.ensure_capacity(KVDemand.same_tokens(allocator, num_tokens))

    def create_prefill_budget(self, allocator, tree_cache, num_mixed_decode_tokens):
        from sglang.srt.mem_cache.prefill_budget import ChainPrefillBudget

        return ChainPrefillBudget(
            allocator, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )


_OWN_POOLS = _OwnPools()
_SHARED_MEMORY = _SharedMemory()


def discipline(allocator):
    """The reclaim discipline of ``allocator``'s pools: the one place their
    declaration (`pools_share_memory`) is read."""
    return _SHARED_MEMORY if allocator.pools_share_memory else _OWN_POOLS


def reclaim_for_prealloc(
    tree_cache: Optional[BasePrefixCache], allocator, demand: KVDemand
) -> Optional[str]:
    """Free room for a preallocation of ``demand``; None means it is ready,
    else a description of the shortfall."""
    return discipline(allocator).reclaim_for_prealloc(tree_cache, allocator, demand)


def admit_preallocation(
    tree_cache: Optional[BasePrefixCache],
    allocator,
    admission: KVDemand,
    take: KVDemand,
    *,
    evictable,
    full_budget_tokens: int,
    window_budget_tokens: Optional[Mapping[str, int]] = None,
) -> bool:
    """Whether a decode-node preallocation is admitted (``admission``, crediting
    what the cache could evict) and its rows (``take``) can be taken now; an
    admission that credits eviction reclaims before the take."""
    return discipline(allocator).admit_preallocation(
        tree_cache,
        allocator,
        admission,
        take,
        evictable=evictable,
        full_budget_tokens=full_budget_tokens,
        window_budget_tokens=window_budget_tokens,
    )


def reclaim_for_load_back(
    tree_cache: Optional[BasePrefixCache], allocator, demand: KVDemand
) -> Optional[str]:
    """Reclaim before a host load takes its rows; None means ready."""
    return discipline(allocator).reclaim_for_load_back(tree_cache, allocator, demand)


def evict_to_free_tokens(
    tree_cache: Optional[BasePrefixCache], allocator, num_tokens: int
) -> Optional[bool]:
    """Evict until ``num_tokens`` new tokens (which every window keeps too) can
    be taken; whether they can, or None with no prefix cache or no answer."""
    return discipline(allocator).evict_to_free_tokens(tree_cache, allocator, num_tokens)


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
    return discipline(allocator).check_decode_capacity(
        tree_cache,
        allocator,
        num_tokens=num_tokens,
        requests=requests,
        spec_algorithm=spec_algorithm,
    )


def create_prefill_budget(
    allocator, tree_cache: Optional[BasePrefixCache], *, num_mixed_decode_tokens=0
):
    """The prefill admission budget over ``allocator`` and ``tree_cache``."""
    return discipline(allocator).create_prefill_budget(
        allocator, tree_cache, num_mixed_decode_tokens
    )


def evict_for_alloc(cache: BasePrefixCache, demand: KVDemand) -> EvictResult:
    """Evict from ``cache`` so that an allocation of ``demand`` (what it will
    take, not what to evict) can be served, by its pool's discipline."""
    return discipline(cache.token_to_kv_pool_allocator).evict_for_alloc(cache, demand)


def windows_share_owner_memory(allocator) -> bool:
    """Whether ``allocator``'s windows take their rows from the owner's memory,
    so a budget over the owner prices them too."""
    return discipline(allocator).windows_share_owner_memory(allocator)
