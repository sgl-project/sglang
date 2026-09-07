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
from typing import Container, Iterable, Sequence

# ~131K entries; at ~150-250 B/entry this is <= ~30 MB and covers roughly
# 8M KV tokens at page size 64 (aux-pool entries included). Coverage per MB
# scales with page size — small-page configs simply remember fewer tokens.
HICACHE_EXISTENCE_CACHE_MAX_ENTRIES = 128 * 1024


class StorageExistenceCache:
    def __init__(self, max_entries: int = HICACHE_EXISTENCE_CACHE_MAX_ENTRIES):
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, pool: str, hashes: Iterable[str]) -> None:
        entries = self._entries
        for h in hashes:
            entries[(pool, h)] = None
            entries.move_to_end((pool, h))
        while len(entries) > self.max_entries:
            entries.popitem(last=False)

    def contains(self, pool: str, page_hash: str) -> bool:
        entries = self._entries
        if (pool, page_hash) not in entries:
            return False
        entries.move_to_end((pool, page_hash))
        return True

    def contains_all(self, pool: str, hashes: Iterable[str]) -> bool:
        return all(self.contains(pool, h) for h in hashes)

    def covers_all(
        self,
        pool: str,
        hashes: Iterable[str],
        extra_cover: Container[str] = frozenset(),
    ) -> bool:
        """True when every page is believed stored or sits in
        ``extra_cover`` (e.g. content past its D2H launch, which always
        reaches its storage-ack). LRU-touches the believed entries."""
        return all(self.contains(pool, h) or h in extra_cover for h in hashes)

    def invalidate_beyond(
        self, pool: str, hashes: Sequence[str], keep_pages: int
    ) -> None:
        """Ground-truth heal from a prefetch hit query: discard beliefs
        beyond the leading ``keep_pages`` of a hash chain (the folded
        usable cut). The next insert re-writes the discarded span, closing
        stale positives and aux holes at the cut."""
        for h in hashes[keep_pages:]:
            self._entries.pop((pool, h), None)

    def clear(self) -> None:
        self._entries.clear()
