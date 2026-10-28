"""Local existence beliefs for HiCache buffer_only mode.

In buffer mode host memory holds no persistent copy, so without a local
existence signal every re-insert of a hot prefix re-stages and re-writes to
L3 storage. The beliefs are that signal: per storage pool, a bounded LRU of
page hashes *believed* present in storage, kept and healed independently so
a node is written pool by pool (only the pools not believed stored).

Semantics are advisory, not authoritative:

- A hit skips the pool's redundant D2H staging + storage write.
- A stale positive (backend evicted the data) costs skipped write-backs until
  a prefetch hit-query shortfall invalidates the pool's entries; the next
  insert then writes that pool back. Never a correctness issue — at worst
  one cold recompute, the same as any cache miss.
- A miss (entry LRU-evicted or never seen) just costs one redundant write
  (idempotent: storage keys are content-addressed).

Keys are the content-chained page hashes already computed at insert time, so
lookups never hash anything and entries survive node deletion, splits, and
recompute (same tokens => same chain). Page hashes are chained and the write
path is prefix-contiguous (parent-cover gate), so a node's own page set is
the only thing a caller needs to check.

A pool also tracks its in-flight content (D2H launch to storage ack), which
admission treats as covered; its ``PoolBeliefPolicy`` says how its keys sit
on the chain (a component may supply one via ``buffer_belief_policy``).

TP determinism: replicas stay identical because every mutation happens on the
scheduler thread at lockstep points with cross-rank-reduced inputs
(storage-ack drain, prefetch-hit drain, fill commit). Do not touch it from
anywhere else.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy

# Per pool: ~524K entries; at ~150-250 B/entry this is <= ~125 MiB and covers
# roughly 32M tokens at page size 64.
HICACHE_EXISTENCE_CACHE_MAX_ENTRIES = 512 * 1024


class PoolExistenceBeliefs:
    """One pool's LRU of page hashes believed stored, plus its refcounted
    in-flight content (D2H launch to storage ack)."""

    def __init__(self, max_entries: int = HICACHE_EXISTENCE_CACHE_MAX_ENTRIES):
        self.max_entries = max_entries
        self._entries: OrderedDict[str, None] = OrderedDict()
        self._inflight: dict[str, int] = {}

    def add(self, hashes: Iterable[str]) -> None:
        entries = self._entries
        move_to_end = entries.move_to_end
        for h in hashes:
            entries[h] = None
            move_to_end(h)
        pop_oldest = entries.popitem
        for _ in range(len(entries) - self.max_entries):
            pop_oldest(last=False)

    def contains(self, page_hash: str) -> bool:
        entries = self._entries
        if page_hash not in entries:
            return False
        entries.move_to_end(page_hash)
        return True

    def contains_all(self, hashes: Iterable[str]) -> bool:
        entries = self._entries
        move_to_end = entries.move_to_end
        for h in hashes:
            try:
                move_to_end(h)
            except KeyError:
                return False
        return True

    def covered(self, hashes: Iterable[str]) -> bool:
        """Every page believed stored or in flight; LRU-touches believed entries."""
        inflight = self._inflight
        if not inflight:
            return self.contains_all(hashes)
        entries = self._entries
        move_to_end = entries.move_to_end
        for h in hashes:
            if h in entries:
                move_to_end(h)
            elif h not in inflight:
                return False
        return True

    def track_inflight(self, hashes: Iterable[str]) -> None:
        """One ref per page hash at its D2H launch (several launched writes
        can carry the same content)."""
        refs = self._inflight
        for h in hashes:
            refs[h] = refs.get(h, 0) + 1

    def untrack_inflight(self, hashes: Iterable[str]) -> None:
        """Drop one ref per page hash at the storage ack (or a failed launch)."""
        refs = self._inflight
        for h in hashes:
            n = refs.get(h, 0) - 1
            if n <= 0:
                refs.pop(h, None)
            else:
                refs[h] = n

    def invalidate_beyond(self, hashes: Sequence[str], keep_pages: int) -> None:
        """Discard the beliefs beyond the leading ``keep_pages`` of ``hashes``;
        the next insert re-writes that span."""
        pop = self._entries.pop
        for h in hashes[keep_pages:]:
            pop(h, None)

    def clear(self) -> None:
        self._entries.clear()
        self._inflight.clear()

    def clear_inflight(self) -> None:
        self._inflight.clear()


class StorageExistenceCache:
    """The beliefs of every storage pool, created on first use."""

    def __init__(self, max_entries: int = HICACHE_EXISTENCE_CACHE_MAX_ENTRIES):
        self.max_entries = max_entries
        self._pools: dict[str, PoolExistenceBeliefs] = {}

    def pool(self, name: str) -> PoolExistenceBeliefs:
        beliefs = self._pools.get(name)
        if beliefs is None:
            beliefs = self._pools[name] = PoolExistenceBeliefs(self.max_entries)
        return beliefs

    def clear(self) -> None:
        for beliefs in self._pools.values():
            beliefs.clear()

    def clear_inflight(self) -> None:
        for beliefs in self._pools.values():
            beliefs.clear_inflight()


class PoolBeliefPolicy:
    """How a pool's keys sit on a node's page-hash chain, hence how a storage
    verdict heals its beliefs: a rank-agreed hit-query boundary (``evidence``
    = the query reported this pool) or a transfer's delivered page count."""

    def heal_from_query(
        self,
        beliefs: PoolExistenceBeliefs,
        chain: Sequence[str],
        keys: Sequence[str],
        hit_pages: int,
        *,
        evidence: bool,
    ) -> None:
        raise NotImplementedError

    def heal_from_transfer(
        self,
        beliefs: PoolExistenceBeliefs,
        keys: Sequence[str],
        delivered_pages: int,
    ) -> None:
        beliefs.invalidate_beyond(keys, delivered_pages)


class ChainPagesPolicy(PoolBeliefPolicy):
    """One key per chain page (ALL_PAGES). The agreed boundary is the MIN of
    per-rank prefix hits, so every rank verified the chain up to it."""

    def heal_from_query(self, beliefs, chain, keys, hit_pages, *, evidence):
        beliefs.invalidate_beyond(chain, hit_pages)
        if evidence:
            beliefs.add(chain[:hit_pages])


class TrailingWindowPolicy(PoolBeliefPolicy):
    """Keys are a trailing window of the chain (TRAILING_PAGES). A rank only
    verified the window ending at its own boundary, so the query only
    invalidates; acks and fills teach presence."""

    def heal_from_query(self, beliefs, chain, keys, hit_pages, *, evidence):
        beliefs.invalidate_beyond(chain, hit_pages)


def default_belief_policy(hit_policy: PoolHitPolicy) -> PoolBeliefPolicy:
    """The built-in policy for a pool no component claims."""
    if hit_policy == PoolHitPolicy.TRAILING_PAGES:
        return TrailingWindowPolicy()
    return ChainPagesPolicy()
