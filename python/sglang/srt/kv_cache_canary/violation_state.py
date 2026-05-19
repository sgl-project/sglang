"""Global violation sink and per-launch health counters shared by all canary launches on one runner."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.jit_kernel.kv_cache_canary_verify import VIOLATION_FIELDS


@dataclass(frozen=True, slots=True, kw_only=True)
class ViolationLog:
    """Global violation sink shared across all canary launches.

    One instance per canary runner — every launch (head / tail / sweep, K / V half, FULL / SWA group) writes
    into the same ring. The kernel_kind field stamped into each violation row identifies which launch fired
    (kernel_kind is a unique int per (head|tail|sweep, K|V, FULL|SWA) tuple; assigned by the runner).

    Ring capacity is sized generously (≥ 1024) so overflow is a non-concern in practice — violations are
    cold-path and the host raises at the first one anyway. atomicAdd contention on a single counter is also
    negligible since violation events are rare.

    Derived state (host computes on read; not stored):
        is_errored = violation_write_index[0] > 0
        first_violation = violation_ring[0]   (valid iff is_errored)
        ring_valid_count = min(violation_write_index[0], ring_capacity)

    The ring is fill-once: writes beyond ring_capacity are dropped but the counter still increments. Whoever
    wins atomicAdd for idx == 0 permanently occupies row 0.

    Fields:
        violation_ring: Append-only violation sink, shape [ring_capacity, VIOLATION_FIELDS], int64. Row 0 is
            the first violation; rows 1..min(write_index, capacity) follow in atomic order. Fill-once.
        violation_write_index: Monotonic violation counter, shape [1], int32. Incremented on every violation
            regardless of ring capacity.
    """

    violation_ring: torch.Tensor
    violation_write_index: torch.Tensor

    @classmethod
    def allocate(cls, *, ring_capacity: int, device: torch.device) -> "ViolationLog":
        if ring_capacity <= 0:
            raise ValueError(
                f"kv-canary: ViolationLog ring_capacity must be positive, got {ring_capacity}"
            )
        return cls(
            violation_ring=torch.zeros(
                ring_capacity, VIOLATION_FIELDS, dtype=torch.int64, device=device
            ),
            violation_write_index=torch.zeros(1, dtype=torch.int32, device=device),
        )

    def clear(self) -> None:
        """Reset to all-zero (forgets all past violations)."""
        self.violation_ring.zero_()
        self.violation_write_index.zero_()
