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

A shared memory has no scalar availability a decision can use: sub-pools take
each other's bytes, and a demand mixes per-token and per-request units in
proportions that differ by request. Its decisions ask a concrete demand
(`capacity_decision` marks them); its scalar reads are estimates for observers,
and reading one inside a decision is an invariant violation (`observe_only_read`).
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import TYPE_CHECKING, Callable, Mapping, Optional

from sglang.srt.mem_cache.kv.alloc.demand import KVDemand

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictResult

logger = logging.getLogger(__name__)


class _DecisionScope(threading.local):
    depth = 0
    warned: frozenset = frozenset()


_decision_scope = _DecisionScope()


def capacity_decision(fn):
    """Mark ``fn`` as a capacity decision: reading a shared-memory pool's
    availability estimate while it runs is a violation (`observe_only_read`)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _decision_scope.depth += 1
        try:
            return fn(*args, **kwargs)
        finally:
            _decision_scope.depth -= 1

    return wrapper


def observe_only_read(what: str) -> None:
    """Called by every availability estimate of a shared-memory pool: inside a
    capacity decision it raises under strict invariant checks, and logs under warn."""
    if not _decision_scope.depth:
        return
    from sglang.srt.environ import InvariantCheckLevel
    from sglang.srt.utils.invariants import resolve_level

    level = resolve_level()
    message = (
        f"{what} is an observe-only estimate of a pool whose sub-pools share "
        "memory, read inside a capacity decision; ask a demand instead "
        "(ChainAllocator.admits, units_fit, or sglang.srt.mem_cache.capacity)"
    )
    if level >= InvariantCheckLevel.STRICT:
        raise AssertionError(message)
    if level >= InvariantCheckLevel.WARN and what not in _decision_scope.warned:
        _decision_scope.warned = _decision_scope.warned | {what}
        logger.warning(message)


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

    shares_memory = False

    def tokens_fit(self, allocator, num_tokens):
        return allocator.available_size() >= num_tokens

    def owner_rows_ready(self, tree_cache, allocator, full_tokens, available):
        if available() < full_tokens:
            tree_cache.evict_for_alloc(KVDemand(full_tokens=full_tokens))
            return available() >= full_tokens
        return True

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

    shares_memory = True

    def owner_rows_ready(self, tree_cache, allocator, full_tokens, available):
        demand = KVDemand(full_tokens=full_tokens).completed(
            allocator.window_pool_names()
        )
        return self.reclaim_for_load_back(tree_cache, allocator, demand) is None

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

    def tokens_fit(self, allocator, num_tokens):
        demand = KVDemand.same_tokens(allocator, num_tokens)
        return allocator.units_fit(
            {n: d for n, d in allocator.page_demands(demand).items() if d > 0}
        )

    def evict_to_free_tokens(self, tree_cache, allocator, num_tokens):
        if not _shares_prefixes(tree_cache):
            return None
        _evict(tree_cache, allocator, KVDemand.same_tokens(allocator, num_tokens))
        return self.tokens_fit(allocator, num_tokens)

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


def shares_memory(allocator) -> bool:
    """Whether ``allocator``'s sub-pools share one memory, so every capacity
    decision asks it a demand."""
    return discipline(allocator).shares_memory


@capacity_decision
def owner_rows_ready(
    tree_cache: BasePrefixCache,
    allocator,
    full_tokens: int,
    available: Callable[[], int],
) -> bool:
    """Evict until a host load's ``full_tokens`` token rows can be taken: pools of
    their own compare ``available``, a shared memory asks the demand itself."""
    return discipline(allocator).owner_rows_ready(
        tree_cache, allocator, full_tokens, available
    )


@capacity_decision
def reclaim_for_prealloc(
    tree_cache: Optional[BasePrefixCache], allocator, demand: KVDemand
) -> Optional[str]:
    """Free room for a preallocation of ``demand``; None means it is ready,
    else a description of the shortfall."""
    return discipline(allocator).reclaim_for_prealloc(tree_cache, allocator, demand)


@capacity_decision
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


@capacity_decision
def reclaim_for_load_back(
    tree_cache: Optional[BasePrefixCache], allocator, demand: KVDemand
) -> Optional[str]:
    """Reclaim before a host load takes its rows; None means ready."""
    return discipline(allocator).reclaim_for_load_back(tree_cache, allocator, demand)


@capacity_decision
def tokens_fit(allocator, num_tokens: int) -> bool:
    """Whether ``num_tokens`` new tokens (which every window keeps too) can be
    taken now, without evicting."""
    return discipline(allocator).tokens_fit(allocator, num_tokens)


@capacity_decision
def evict_to_free_tokens(
    tree_cache: Optional[BasePrefixCache], allocator, num_tokens: int
) -> Optional[bool]:
    """Evict until ``num_tokens`` new tokens (which every window keeps too) can
    be taken; whether they can, or None with no prefix cache or no answer."""
    return discipline(allocator).evict_to_free_tokens(tree_cache, allocator, num_tokens)


@capacity_decision
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


@capacity_decision
def create_prefill_budget(
    allocator, tree_cache: Optional[BasePrefixCache], *, num_mixed_decode_tokens=0
):
    """The prefill admission budget over ``allocator`` and ``tree_cache``."""
    return discipline(allocator).create_prefill_budget(
        allocator, tree_cache, num_mixed_decode_tokens
    )


@capacity_decision
def evict_for_alloc(cache: BasePrefixCache, demand: KVDemand) -> EvictResult:
    """Evict from ``cache`` so that an allocation of ``demand`` (what it will
    take, not what to evict) can be served, by its pool's discipline."""
    return discipline(cache.token_to_kv_pool_allocator).evict_for_alloc(cache, demand)


def windows_share_owner_memory(allocator) -> bool:
    """Whether ``allocator``'s windows take their rows from the owner's memory,
    so a budget over the owner prices them too."""
    return discipline(allocator).windows_share_owner_memory(allocator)
