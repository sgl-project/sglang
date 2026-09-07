"""Pool sides of the unified-memory composites: one `MultiEndedKVPool`
each, sharing a virtual id space with their peers."""

from __future__ import annotations

from typing import List, Optional

import torch

from sglang.srt.mem_cache.allocator.base import (
    BaseKVPoolSide,
    KVPoolSide,
    invariant_checks_enabled,
)
from sglang.srt.mem_cache.allocator.unified_sub_pool import MultiEndedKVPool
from sglang.srt.utils.invariants import Bucket, Invariant, IsTrue, expect

# A tombstoned (-1) or sink (0) page handed to the swa free would join the free
# list and be handed out twice.
_SWA_VIRTUAL_LIVE = Invariant("swa.virtual_live", Bucket.FATAL_UNCONTAINABLE, IsTrue())


class VirtualFullKVPoolSide(KVPoolSide):
    """The id-owning full sub-pool. `conserve_cap` is the static partition cap
    the leak invariant measures against."""

    pool: MultiEndedKVPool

    def __init__(self, pool: MultiEndedKVPool, *, conserve_cap: int):
        super().__init__(pool)
        self.conserve_cap = conserve_cap

    def available_size(self) -> int:
        # The static-conserve cap bounds a side lending bytes to its peer, the
        # byte-coordinated view bounds one that has grown into the shared gap;
        # whichever is tighter is what a scheduler may still allocate.
        return min(self.conserve_available_size(), self.schedulable_available_size())

    def conserve_available_size(self) -> int:
        return self.conserve_cap - self.pool.allocated_count()

    def free(self, free_index: torch.Tensor) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        self.pool.free(free_index.detach().to(torch.int64))
        if self.pool.free_group is None:
            self.pool.clear_inverse_history()

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        self.pool.free_segment(free_index.detach().to(torch.int64), start_pos=start_pos)
        if self.pool.free_group is None:
            self.pool.clear_inverse_history()

    def free_group_end(self) -> None:
        self.pool.free_group_end()
        self.pool.clear_inverse_history()


class VirtualSWAKVPoolSide(BaseKVPoolSide):
    """The non-owner swa sub-pool, addressed by the shared virtual ids. Every id
    freed here is live on the swa side (the release contract), so the pool's
    v2p table is never consulted before a free."""

    def __init__(self, pool: MultiEndedKVPool, *, conserve_cap: int):
        self.pool = pool
        self.page_size = pool.page_size
        self.conserve_cap = conserve_cap
        # Token ids and page representatives are buffered separately so the
        # flush keeps the fixed-shape page path.
        self.free_group = None
        self._pending_reps: Optional[List[torch.Tensor]] = None

    def available_size(self) -> int:
        return min(self.conserve_available_size(), self.schedulable_available_size())

    def conserve_available_size(self) -> int:
        return self.conserve_cap - self.pool.allocated_count()

    def schedulable_available_size(self) -> int:
        return self.pool.schedulable_available_size()

    def free(self, free_index: torch.Tensor) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        v = free_index.detach().to(torch.int64)
        if self.free_group is not None:
            self.free_group.append(self._copy_for_free_group(v))
            return
        self._release_tokens(v)

    def _expect_live(self, pages: torch.Tensor) -> None:
        if invariant_checks_enabled():
            expect(
                _SWA_VIRTUAL_LIVE,
                self.pool.virtual_to_physical[pages] > 0,
                msg="swa side already released; caller wants full.free",
            )

    def _release_tokens(self, v: torch.Tensor) -> None:
        self._expect_live(v // self.page_size)
        if self.page_size == 1:
            # token == page: the ids are already unique page ids.
            self.pool.free(v, _pages=v)
        else:
            self.pool.free(v)
        self.pool.clear_inverse_history()

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        if self.page_size == 1:
            self.free(free_index)
            return
        reps = self.pool._page_reps(free_index.detach().to(torch.int64), start_pos)
        if self._pending_reps is not None:
            self._pending_reps.append(reps)
            return
        self._release_reps(reps)

    def _release_reps(self, reps: torch.Tensor) -> None:
        self._expect_live(reps // self.page_size)
        self.pool.free(reps, _pages=reps // self.page_size)
        self.pool.clear_inverse_history()

    def free_group_begin(self) -> None:
        super().free_group_begin()
        self._pending_reps = []

    def free_group_end(self) -> None:
        pending, self.free_group = self.free_group, None
        reps, self._pending_reps = self._pending_reps, None
        if pending:
            self._release_tokens(torch.cat(pending))
        if reps:
            self._release_reps(torch.cat(reps))
