"""Mamba overflow allocator — reserved-row ring inside MambaPoolHost.

Background
----------
Under the post-merge hybrid cache controller (`hybrid_cache_controller.py`),
write_backup auto-allocates a host slot per ``PoolTransfer`` via
``_resolve_pool_transfers_allocation``. If the mamba host pool is full and
eviction cannot free a slot, the controller rolls back the entire write batch
(including the KV side) and returns ``None``.

The pre-merge code path silently dropped just the mamba companion and let the
KV write proceed, which produced the headline failure mode in
``.planning/quick/20260503-l3-mamba-companion-recompute/PROPOSAL.md`` (companion
coverage 0.032%, ``LRUHiCacheFile get_hit=0``). Under the post-merge controller
the same condition now also kills the KV write — a strictly worse outcome.

Alt B (this module) fixes both: a small ring of *reserved rows* lives at the
end of the normal ``MambaPoolHost`` buffers. Those rows are never seen by the
LRU allocator (``free_slots`` is initialised over ``[0, size)`` only), so they
cannot be evicted. ``_MambaOverflowAllocator`` hands them out via ref-counted
``acquire()`` / ``release(slot_idx)`` calls, with absolute slot indices in the
range ``[base, base + size)``.

Because slots are just integer indices into the pool's tensors, every
downstream consumer (D→H copy in
``MambaPoolHost.backup_from_device_all_layer``, storage write via
``HiCacheFile._write_page`` →  ``host_pool.get_data_page(slot_idx, flat=True)``)
treats them identically to regular slots. No new code path is required in
``hicache_storage.py`` or the ``HostPoolGroup`` dispatch layer.

Design decisions
----------------
- **Fixed size, no resize.** Pre-allocated at construction so we never trip
  ``cudaHostAlloc`` latency spikes during writeback.
- **Ref-counted, not single-shot.** A slot stays alive across the async
  D→H→storage pipeline until the H→S archive completes; the H→S commit hook
  releases the slot back to the ring.
- **Round-robin acquire.** Spreads wear across all reserved rows so the same
  pinned page isn't repeatedly written.
- **Thread-safe.** ``threading.Lock`` around all bookkeeping. ``write_backup``
  fires on the scheduler thread; archive commit fires from the backup-drain
  thread.
- **Index-agnostic.** The allocator does NOT own the underlying tensors. It
  only owns the integer slot-id allocation policy. The mamba host pool owns
  the bytes. This keeps a clean separation and means changes to the pool's
  on-disk layout don't ripple into the allocator.

User-tunable via ``--mamba-overflow-size N`` (default 8). At 8 slots × ~2 MB
per slot on Qwen3.6-27B-FP8, total reserved memory is ~16 MB — negligible vs
the multi-GB mamba host pool and the 10 GiB ``HICACHE_HOST_MEMORY_RESERVE``.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class _MambaOverflowAllocator:
    """Ref-counted ring allocator for reserved rows inside ``MambaPoolHost``.

    Owns a half-open index range ``[base, base + size)``. Tracks per-slot
    refcounts so the same row stays alive across the async D→H→storage
    pipeline.

    Public API
    ----------
    acquire() -> Optional[int]
        Returns an absolute slot index (in ``[base, base+size)``) with
        refcount = 1, or ``None`` if every slot is currently in use.
    retain(slot_idx) -> None
        Increment refcount on an in-use slot (used when one D→H landed slot
        is referenced by multiple downstream operations).
    release(slot_idx) -> None
        Decrement refcount; when it hits zero the slot becomes acquirable
        again. Raises if called on a slot whose refcount is already zero.
    stats() -> dict
        Telemetry snapshot. Emitted as a ``MambaOverflowRing stats: ...``
        log line every ~200 ops by ``MambaPoolHost.overflow_stats_log_if_due``.

    Slot indices are absolute (``base + i`` for the i-th reserved row). The
    rest of the system never sees the relative form; this keeps the rest of
    the codebase index-agnostic — a ``PoolTransfer.host_indices`` tensor
    containing slot index 12345 looks identical whether 12345 came from the
    normal allocator or the overflow ring.
    """

    LOG_EVERY = 200

    def __init__(self, base_idx: int, size: int):
        if size < 0:
            raise ValueError(f"overflow size must be >= 0, got {size}")
        self._base = int(base_idx)
        self._size = int(size)
        self._refcounts: list[int] = [0] * self._size
        self._next_scan = 0
        self._lock = threading.Lock()
        self._stats = {
            "acquires": 0,
            "releases": 0,
            "retains": 0,
            "denials": 0,
            "in_use_peak": 0,
            "double_release": 0,
        }
        self._ops_since_log = 0

    @property
    def base(self) -> int:
        return self._base

    @property
    def size(self) -> int:
        return self._size

    def contains(self, slot_idx: int) -> bool:
        """Whether ``slot_idx`` is owned by this allocator."""
        return self._base <= slot_idx < self._base + self._size

    def acquire(self) -> Optional[int]:
        if self._size == 0:
            return None
        with self._lock:
            for offset in range(self._size):
                rel = (self._next_scan + offset) % self._size
                if self._refcounts[rel] == 0:
                    self._refcounts[rel] = 1
                    self._next_scan = (rel + 1) % self._size
                    self._stats["acquires"] += 1
                    in_use = sum(1 for c in self._refcounts if c > 0)
                    if in_use > self._stats["in_use_peak"]:
                        self._stats["in_use_peak"] = in_use
                    self._ops_since_log += 1
                    return self._base + rel
            self._stats["denials"] += 1
            self._ops_since_log += 1
            return None

    def retain(self, slot_idx: int) -> None:
        rel = self._rel(slot_idx)
        with self._lock:
            if self._refcounts[rel] == 0:
                raise RuntimeError(
                    f"_MambaOverflowAllocator.retain on free slot {slot_idx} "
                    f"(rel={rel})"
                )
            self._refcounts[rel] += 1
            self._stats["retains"] += 1

    def release(self, slot_idx: int) -> None:
        rel = self._rel(slot_idx)
        with self._lock:
            if self._refcounts[rel] == 0:
                # Defensive: log and count, do NOT crash the scheduler thread.
                # A double-release indicates a wiring bug but should not abort
                # in-flight requests.
                self._stats["double_release"] += 1
                logger.warning(
                    "_MambaOverflowAllocator double-release on slot %d (rel=%d). "
                    "stats=%s",
                    slot_idx,
                    rel,
                    dict(self._stats),
                )
                return
            self._refcounts[rel] -= 1
            self._stats["releases"] += 1
            self._ops_since_log += 1

    def _rel(self, slot_idx: int) -> int:
        if not self.contains(slot_idx):
            raise ValueError(
                f"slot {slot_idx} not in this allocator's range "
                f"[{self._base}, {self._base + self._size})"
            )
        return slot_idx - self._base

    def stats(self) -> dict:
        with self._lock:
            snapshot = dict(self._stats)
            snapshot["in_use_now"] = sum(1 for c in self._refcounts if c > 0)
            snapshot["base"] = self._base
            snapshot["size"] = self._size
        return snapshot

    def log_if_due(self) -> None:
        """Emit ``MambaOverflowRing stats: ...`` every LOG_EVERY ops.

        Called from ``MambaPoolHost.overflow_acquire`` / ``overflow_release``
        wrappers. The cadence matches ``LRUHiCacheFile`` so the two stat
        streams interleave at similar granularity.
        """
        with self._lock:
            if self._ops_since_log < self.LOG_EVERY:
                return
            self._ops_since_log = 0
            snapshot = dict(self._stats)
            snapshot["in_use_now"] = sum(1 for c in self._refcounts if c > 0)
        logger.info("MambaOverflowRing stats: %s", snapshot)
