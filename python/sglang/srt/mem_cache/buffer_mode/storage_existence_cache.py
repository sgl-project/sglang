"""Local existence cache for HiCache buffer_only mode.

In buffer mode host memory holds no persistent copy, so without a local
existence signal every re-insert of a hot prefix re-stages and re-writes to
L3 storage. This cache is that signal: a bounded LRU of (pool, page-hash)
entries *believed* present in storage.

Semantics are advisory, not authoritative:

- A hit skips the redundant D2H + storage write.
- A stale positive (backend evicted the data) costs skipped write-backs until
  a prefetch hit-query shortfall invalidates the entries; the next insert
  then writes the data back. Never a correctness issue — at worst one cold
  recompute, the same as any cache miss.
- A miss (entry LRU-evicted or never seen) just costs one redundant write
  (idempotent: storage keys are content-addressed).

Keys are the content-chained page hashes already computed at insert time, so
lookups never hash anything and entries survive node deletion, splits, and
recompute (same tokens => same chain). Page hashes are chained and the write
path is prefix-contiguous (parent-cover gate), so a node's own page set is
the only thing a caller needs to check.

TP determinism: replicas stay identical because every mutation happens on the
scheduler thread at lockstep points with cross-rank-reduced inputs
(storage-ack drain, prefetch-hit drain, fill commit). Do not touch it from
anywhere else.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Container, Iterable, Sequence

# ~524K entries; at ~150-250 B/entry this is <= ~125 MiB and covers roughly
# 32M tokens at page size 64 across all pools.
HICACHE_EXISTENCE_CACHE_MAX_ENTRIES = 512 * 1024


class StorageExistenceCache:
    def __init__(self, max_entries: int = HICACHE_EXISTENCE_CACHE_MAX_ENTRIES):
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, pool: str, hashes: Iterable[str]) -> None:
        entries = self._entries
        move_to_end = entries.move_to_end
        for h in hashes:
            key = (pool, h)
            entries[key] = None
            move_to_end(key)
        pop_oldest = entries.popitem
        for _ in range(len(entries) - self.max_entries):
            pop_oldest(last=False)

    def contains(self, pool: str, page_hash: str) -> bool:
        entries = self._entries
        key = (pool, page_hash)
        if key not in entries:
            return False
        entries.move_to_end(key)
        return True

    def contains_all(self, pool: str, hashes: Iterable[str]) -> bool:
        entries = self._entries
        move_to_end = entries.move_to_end
        for h in hashes:
            key = (pool, h)
            try:
                move_to_end(key)
            except KeyError:
                return False
        return True

    def covers_all(
        self,
        pool: str,
        hashes: Iterable[str],
        extra_cover: Container[str] = frozenset(),
    ) -> bool:
        """True when every page is believed stored or sits in
        ``extra_cover`` (e.g. content past its D2H launch, which always
        reaches its storage-ack). LRU-touches the believed entries."""
        entries = self._entries
        move_to_end = entries.move_to_end
        if extra_cover:
            for h in hashes:
                key = (pool, h)
                if key in entries:
                    move_to_end(key)
                elif h not in extra_cover:
                    return False
            return True
        return self.contains_all(pool, hashes)

    def invalidate_beyond(
        self, pool: str, hashes: Sequence[str], keep_pages: int
    ) -> None:
        """Ground-truth heal from a prefetch hit query: discard beliefs
        beyond the leading ``keep_pages`` of a hash chain (the folded
        usable cut). The next insert re-writes the discarded span, closing
        stale positives and aux holes at the cut."""
        pop = self._entries.pop
        for h in hashes[keep_pages:]:
            pop((pool, h), None)

    def clear(self) -> None:
        self._entries.clear()
