"""Refcounted content-addressed page cache over the buffer-mode host pool.

NOT WIRED YET: staged spans register as ``(pool, page_hash) -> (slots,
refcount)`` so prefetches can be served zero-copy from local staging
(write-around / promote-on-read retention, zero-ref LRU reclaim); keys are
the content-chained page hashes, so entries survive node deletion, splits,
and recompute. Counterpart of ``StorageExistenceCache`` (beliefs about
STORAGE, dedupes writes); this tracks LOCAL HOST RAM and dedupes loads.

TP determinism: replicas must stay identical across attention ranks — the
cache feeds scheduler-visible structure, so divergence is a collective
hang, not a soft miss. Preconditions when wiring: (1) mutate only on the
scheduler thread at lockstep points; (2) rank-reduce any fold anchored by a
per-rank storage outcome (hit count, revoke) before it picks a mutation;
(3) controller queues stay FIFO and single-threaded so MIN-count drains
process the same prefix on every rank.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)


class _PageRef:
    """One cached page: host slot span, reader refcount, and whether to
    retain at refs==0 (write-around: write staging frees at its storage ack
    unless a read promoted it). ``first_slot`` mirrors ``slots[0]`` as a
    plain int because per-page tensor-scalar reads are too slow in
    ``release``."""

    __slots__ = ("slots", "first_slot", "refs", "retain")

    def __init__(
        self, slots: torch.Tensor, first_slot: int, refs: int = 1, retain: bool = True
    ):
        self.slots = slots
        self.first_slot = first_slot
        self.refs = refs
        self.retain = retain


class BufferPageCache:
    def __init__(self) -> None:
        # (pool, page_hash) -> _PageRef; slots stay allocated in the host
        # pool for as long as the entry exists.
        self._entries: dict[tuple[str, str], _PageRef] = {}
        # Per-pool zero-ref LRU (head = coldest): reclaim victims.
        self._zero_ref: dict[str, OrderedDict[str, None]] = {}
        # Per-pool slot tokens held by the cache (refed + zero-ref).
        self._held_tokens: dict[str, int] = {}
        # Per-pool slot tokens at refs=0 (reclaimable under pressure).
        self._zero_ref_tokens: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def num_zero_ref_pages(self) -> int:
        return sum(len(lru) for lru in self._zero_ref.values())

    def held_tokens(self, pool: str) -> int:
        return self._held_tokens.get(pool, 0)

    def zero_ref_tokens(self, pool: str) -> int:
        """Slot tokens reclaimable right now (zero-ref cached pages).
        Occupancy/rate-limit gates must treat these as free-able, not used:
        a pool full of zero-ref cache is one reclaim away from empty."""
        return self._zero_ref_tokens.get(pool, 0)

    def register(
        self,
        pool: str,
        hashes: Sequence[str],
        host_indices: torch.Tensor,
        page_size: int,
        retain: bool = True,
    ) -> int:
        """Cache a staged span, one entry per page, refs=1 (the staging op);
        returns the number of pages newly cached. ``retain=False`` =
        write-around (freed at last ref unless a read hit promotes it); a
        duplicate hash keeps the existing entry and the newcomer's slots
        stay op-owned for a raw free at release."""
        assert len(host_indices) == len(hashes) * page_size
        registered = 0
        entries = self._entries
        # One batched read of the page-boundary slot ids (see _PageRef).
        first_slots = host_indices[::page_size].tolist()
        for i, page_hash in enumerate(hashes):
            key = (pool, page_hash)
            existing = entries.get(key)
            if existing is not None:
                existing.retain = existing.retain or retain
                continue
            entries[key] = _PageRef(
                host_indices[i * page_size : (i + 1) * page_size],
                first_slots[i],
                retain=retain,
            )
            registered += 1
        if registered:
            self._held_tokens[pool] = (
                self._held_tokens.get(pool, 0) + registered * page_size
            )
        return registered

    def contains(self, pool: str, page_hash: str) -> bool:
        """Non-mutating presence probe (no LRU touch)."""
        return (pool, page_hash) in self._entries

    def peek_run_len(self, pool: str, hashes: Sequence[str]) -> int:
        """Length of the leading run of cached pages. Non-mutating."""
        entries = self._entries
        run = 0
        for page_hash in hashes:
            if (pool, page_hash) not in entries:
                break
            run += 1
        return run

    def acquire(self, pool: str, hashes: Sequence[str]) -> Optional[torch.Tensor]:
        """refs++ on every page and return the gathered slot tensor (pages
        expanded to token slots, in page order). All-or-nothing: returns
        None without mutating if any page is missing."""
        entries = self._entries
        refs = []
        for page_hash in hashes:
            entry = entries.get((pool, page_hash))
            if entry is None:
                return None
            refs.append(entry)
        zero_ref = self._zero_ref.get(pool)
        for page_hash, entry in zip(hashes, refs):
            if entry.refs == 0 and zero_ref is not None:
                if zero_ref.pop(page_hash, None) is not None:
                    self._zero_ref_tokens[pool] -= len(entry.slots)
            entry.refs += 1
            # Read demand proven: promote write-around pages to retained.
            entry.retain = True
        return torch.cat([entry.slots for entry in refs])

    def release(
        self,
        pool: str,
        hashes: Sequence[str],
        host_indices: torch.Tensor,
        page_size: int,
    ) -> Optional[torch.Tensor]:
        """Drop one ref per page of a span: at refs==0 retained pages move
        to the zero-ref LRU tail while write-around pages return their
        slots for an immediate free. Duplicate-staging slots (canonical
        entry lives elsewhere) are returned for a raw free without touching
        the canonical refcount."""
        assert len(host_indices) == len(hashes) * page_size
        leftover: list[torch.Tensor] = []
        entries = self._entries
        # One batched read of the page-boundary slot ids (see _PageRef).
        first_slots = host_indices[::page_size].tolist()
        for i, page_hash in enumerate(hashes):
            entry = entries.get((pool, page_hash))
            if entry is None or entry.first_slot != first_slots[i]:
                leftover.append(host_indices[i * page_size : (i + 1) * page_size])
                continue
            assert entry.refs > 0, "release without a matching acquire/register"
            entry.refs -= 1
            if entry.refs == 0:
                if entry.retain:
                    self._zero_ref.setdefault(pool, OrderedDict())[page_hash] = None
                    self._zero_ref_tokens[pool] = self._zero_ref_tokens.get(
                        pool, 0
                    ) + len(entry.slots)
                else:
                    del entries[(pool, page_hash)]
                    self._held_tokens[pool] -= len(entry.slots)
                    leftover.append(entry.slots)
        if not leftover:
            return None
        return torch.cat(leftover)

    def reclaim(
        self,
        pool: str,
        need_tokens: int,
        free: Callable[[torch.Tensor], int],
    ) -> int:
        """Pop zero-ref LRU heads, free their slots back to the host pool,
        and drop the entries. Called under allocation pressure only. Returns
        the number of slot tokens freed (may undershoot when everything
        left is refed)."""
        zero_ref = self._zero_ref.get(pool)
        if not zero_ref or need_tokens <= 0:
            return 0
        freed = 0
        batch: list[torch.Tensor] = []
        while zero_ref and freed < need_tokens:
            page_hash, _ = zero_ref.popitem(last=False)
            entry = self._entries.pop((pool, page_hash))
            batch.append(entry.slots)
            freed += len(entry.slots)
        if batch:
            free(torch.cat(batch))
            self._held_tokens[pool] -= freed
            self._zero_ref_tokens[pool] -= freed
        return freed


class BufferPageCacheOps:
    """Pool-facing operations over a :class:`BufferPageCache`: span/hold
    registration and release keyed the way the storage write keys them,
    pressure reclaim, and the SWA-folded continuation fold. The caller owns
    the collectives — rank-reduce any fold anchored by a per-rank storage
    outcome before acting on it (see the module docstring)."""

    def __init__(
        self,
        page_cache: BufferPageCache,
        mem_pool_host,
        sw_window_pages_fn: Callable[[], int],
    ):
        # Rebound by the owner when the structure is recreated (reset).
        self.page_cache = page_cache
        self._mem_pool_host = mem_pool_host
        # SWA window in KV pages when SWA stages through a host pool
        # (0 = KV-only: no trailing window in the fold).
        self._sw_window_pages_fn = sw_window_pages_fn

    def aux_window_keys(
        self, hash_values: list[str], transfer: PoolTransfer
    ) -> Optional[list[str]]:
        """Trailing KV page hashes keying an aux transfer's staged window
        (one key per aux-pool page), recomputed from the rank-synced span
        hashes so registration and release always agree across ranks."""
        if transfer.host_indices is None or transfer.host_indices.numel() == 0:
            return None
        if transfer.indices_from_pool is not None:
            return None  # sidecar rides another pool's slots; nothing to key
        entry = self._mem_pool_host.entry_map.get(transfer.name)
        if entry is None:
            return None
        pool_page_size = entry.host_pool.page_size
        num_keys = len(transfer.host_indices) // pool_page_size
        if num_keys == 0 or num_keys > len(hash_values):
            return None
        return hash_values[-num_keys:]

    def register_span(
        self,
        pool: PoolName,
        hashes: list[str],
        host_indices: torch.Tensor,
        retain: bool = True,
    ) -> None:
        """Cache a page-aligned staged span (refs=1 for the staging op)."""
        if not hashes:
            return
        entry = self._mem_pool_host.entry_map.get(pool)
        if entry is None:
            return
        self.page_cache.register(
            pool, hashes, host_indices, entry.host_pool.page_size, retain=retain
        )

    def release_span(
        self,
        pool: PoolName,
        hashes: list[str],
        host_indices: torch.Tensor,
    ) -> None:
        """Drop the staging op's ref on a span; zero-ref pages stay cached
        (servable) until pressure reclaims them. Op-owned duplicate slots
        (their hash was cached elsewhere) are freed raw, as before."""
        if host_indices is None or host_indices.numel() == 0:
            return
        entry = self._mem_pool_host.entry_map.get(pool)
        if entry is None:
            return
        if not hashes:
            entry.host_pool.free(host_indices)
            return
        leftover = self.page_cache.release(
            pool, hashes, host_indices, entry.host_pool.page_size
        )
        if leftover is not None and leftover.numel() > 0:
            entry.host_pool.free(leftover)

    def register_hold(
        self,
        hash_values: list[str],
        host_indices: torch.Tensor,
        aux_xfers: list[PoolTransfer],
        retain: bool = True,
    ) -> None:
        """Register a staged KV span plus its aux windows (SWA/Mamba states
        keyed by their trailing KV page hashes, same keying the storage
        write uses). ``retain=False`` = write-around: servable only while
        the staging op pins the slots, freed at the last release unless a
        read hit promotes it."""
        self.register_span(PoolName.KV, hash_values, host_indices, retain=retain)
        for transfer in aux_xfers:
            keys = self.aux_window_keys(hash_values, transfer)
            if keys is not None:
                self.register_span(
                    transfer.name, keys, transfer.host_indices, retain=retain
                )

    def release_hold(
        self,
        hash_values: list[str],
        host_indices: torch.Tensor,
        aux_xfers: list[PoolTransfer],
    ) -> None:
        """Mirror of register_hold for every hold retirement path
        (storage-ack, fill H2D-ack, staged drop, abort)."""
        self.release_span(PoolName.KV, hash_values, host_indices)
        for transfer in aux_xfers:
            if transfer.indices_from_pool is not None:
                continue
            keys = self.aux_window_keys(hash_values, transfer)
            self.release_span(transfer.name, keys or [], transfer.host_indices)

    def reclaim(self, pool: PoolName, num_tokens: int) -> int:
        """Free just enough zero-ref cached pages for an allocation of
        num_tokens to succeed. Scheduler-thread only (lockstep pressure
        points: staging-hit alloc, prepare_prefetch, cc.write)."""
        entry = self._mem_pool_host.entry_map.get(pool)
        if entry is None:
            return 0
        shortfall = num_tokens - entry.host_pool.available_size()
        if shortfall <= 0:
            return 0
        return self.page_cache.reclaim(pool, shortfall, entry.host_pool.free)

    def continuation_run(self, chain: list[str], start_pages: int) -> int:
        """Longest cached run continuing the span at page ``start_pages``
        (0 = leading run), folded for SWA: the joint span's trailing window
        must be fully cache-servable, mirroring batch_exists_v2's
        trailing_pages fold. Non-mutating and rank-deterministic."""
        page_cache = self.page_cache
        kv_run = page_cache.peek_run_len(PoolName.KV, chain[start_pages:])
        if kv_run == 0:
            return 0
        sw_pages = self._sw_window_pages_fn()
        if sw_pages == 0:
            return kv_run
        for cont in range(kv_run, 0, -1):
            joint = start_pages + cont
            window = min(sw_pages, joint)
            if cont < window:
                # Window straddles into the head; only possible for
                # anchored runs, and shrinking cont cannot fix it.
                break
            if all(
                page_cache.contains(PoolName.SWA, chain[i])
                for i in range(joint - window, joint)
            ):
                return cont
        return 0

    def acquire_span(
        self, chain: list[str], start_pages: int, cont_pages: int
    ) -> Optional[tuple[torch.Tensor, list[PoolTransfer]]]:
        """Acquire a folded continuation run: its KV pages plus the JOINT
        span's trailing SWA window (refs++ on every page). Returns
        (kv_slots, aux_xfers) or None (with no refs held) if a page
        vanished since the fold — defensive; fold and acquire run in the
        same lockstep step."""
        page_cache = self.page_cache
        cont_hashes = list(chain[start_pages : start_pages + cont_pages])
        kv_slots = page_cache.acquire(PoolName.KV, cont_hashes)
        if kv_slots is None:
            return None
        aux_xfers: list[PoolTransfer] = []
        sw_pages = self._sw_window_pages_fn()
        if sw_pages > 0:
            joint = start_pages + cont_pages
            window_hashes = list(chain[joint - min(sw_pages, joint) : joint])
            swa_slots = page_cache.acquire(PoolName.SWA, window_hashes)
            if swa_slots is None:
                self.release_span(PoolName.KV, cont_hashes, kv_slots)
                return None
            aux_xfers.append(
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=swa_slots,
                    keys=window_hashes,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        return kv_slots, aux_xfers
