"""Mirrored per-CP-rank pool availability (DESIGN_kv_reshard.md §7).

Under CP KV-resharding, every CP rank stores only its own slice of the
KV bytes (size ``size_local`` rows per layer). The global "is there
room?" decision must consider per-rank pool occupancy, not a single
shared counter, because over-committing one rank's pool while another
sits idle would block admission even though aggregate capacity exists.

This class tracks the per-rank remaining row budget. Every CP rank
instantiates it with the same args and runs identical operations on
identical inputs (SPMD), so ``local_available`` stays mirrored across
ranks without any allgather/allreduce.
"""

from __future__ import annotations

from typing import List, Sequence


class MirroredCpAvailability:
    def __init__(self, cp_size: int, size_local: int) -> None:
        if cp_size <= 0:
            raise ValueError(f"cp_size must be positive, got {cp_size}")
        if size_local < 0:
            raise ValueError(f"size_local must be non-negative, got {size_local}")
        self.cp_size = cp_size
        self.size_local = size_local
        self.local_available: List[int] = [size_local] * cp_size

    def can_admit(self, owned_counts: Sequence[int]) -> bool:
        """Return True iff every rank has enough room for its owned slice."""
        self._check_len(owned_counts)
        return all(
            self.local_available[r] >= owned_counts[r] for r in range(self.cp_size)
        )

    def alloc(self, owned_counts: Sequence[int]) -> None:
        """Reserve ``owned_counts[r]`` rows from rank ``r``'s budget."""
        self._check_len(owned_counts)
        for r in range(self.cp_size):
            self.local_available[r] -= owned_counts[r]

    def free(self, owned_counts: Sequence[int]) -> None:
        """Return ``owned_counts[r]`` rows to rank ``r``'s budget."""
        self._check_len(owned_counts)
        for r in range(self.cp_size):
            self.local_available[r] += owned_counts[r]

    def min_available(self) -> int:
        """Smallest remaining budget across all ranks. Useful for admission
        sizing where the constraint is the most-loaded rank."""
        return min(self.local_available)

    def _check_len(self, owned_counts: Sequence[int]) -> None:
        if len(owned_counts) != self.cp_size:
            raise ValueError(
                f"owned_counts must have length cp_size={self.cp_size}, "
                f"got length {len(owned_counts)}"
            )

    def __repr__(self) -> str:
        return (
            f"MirroredCpAvailability(cp_size={self.cp_size}, "
            f"size_local={self.size_local}, "
            f"local_available={self.local_available})"
        )
