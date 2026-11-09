# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The slot allocator of a unified pool's state band, as the static slot
protocol presents it (deleted with that protocol's last consumer)."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

from sglang.srt.mem_cache.kv.state_storage import StateStorage

logger = logging.getLogger(__name__)


class UnifiedMambaSlotAllocator:
    """Expose unified state slots through the ``MambaSlotAllocator`` interface.

    ``alloc()`` returns virtual ids; ``translate()`` maps them to physical slots.
    ``StateView.clear_slots`` owns clearing, not allocation.
    """

    def __init__(
        self,
        mea,
        device: str,
        take: Optional[Callable[[int], Optional[torch.Tensor]]] = None,
    ):
        self._multi_ended_allocator = mea
        # Slots are taken through the chain allocator when one owns the band.
        self._take = mea.alloc if take is None else take
        self._device = device
        self._alloc_iter = None  # active alloc_group batch iterator

    # -- translation (owns the v<->p mapping) --

    def translate(self, virtual_ids: torch.Tensor) -> torch.Tensor:
        # VIRTUAL -> PHYSICAL slot ids; page_size==1, so a direct v2p gather.
        return self._multi_ended_allocator.ids.virtual_to_physical[virtual_ids]

    @property
    def virtual_to_physical(self) -> torch.Tensor:
        return self._multi_ended_allocator.ids.virtual_to_physical

    # -- sizing / free-list --

    @property
    def size(self) -> int:
        return self._multi_ended_allocator.max_slots - 1

    def available_size(self) -> int:
        # Slot-conservation count (max - allocated): the leak-check view, NOT the
        # planner value (use schedulable_available_size for that).
        return self.size - self._multi_ended_allocator.allocated_count()

    def schedulable_available_size(self) -> int:
        # Byte-coordinated count (>= N => alloc(N) succeeds); credits the peer's
        # drainable holes since alloc flushes the peer before extending.
        return self._multi_ended_allocator.schedulable_available_size()

    @property
    def free_slots(self) -> torch.Tensor:
        # Watermark-derived physical free-list for the invariant checker.
        a = self._multi_ended_allocator
        assert a.page_size == 1, (
            "UnifiedMambaSlotAllocator.free_slots assumes page_size==1; got "
            f"{a.page_size}. Mamba state is per-request, orthogonal to paging."
        )
        if a.grow_direction == "up":
            start, end = a.watermark_physical, a.num_pages
        else:
            start, end = a.min_page_index, a.watermark_physical + 1
        if start >= end:
            return torch.empty((0,), dtype=torch.int64, device=self._device)
        return torch.arange(start, end, dtype=torch.int64, device=self._device)

    # -- slot management (delegates to the Band) --

    def alloc(self, need_size: int):
        # alloc_group fast path: single-slot draws from the prefetched batch.
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._take(need_size)  # VIRTUAL ids

    def free(self, free_index: torch.Tensor):
        return self._multi_ended_allocator.free(free_index)

    def clear(self):
        self._alloc_iter = None
        return self._multi_ended_allocator.clear()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch that ``alloc(1)`` then draws from."""
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._take(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self._multi_ended_allocator.free(torch.cat(remaining))
        self._alloc_iter = None

    def is_slot_allocated(self, slot) -> bool:
        return self._multi_ended_allocator.is_slot_allocated(int(slot))

    def allocator_state_str(self) -> str:
        return self._multi_ended_allocator.allocator_state_str()


def _wire_mamba_slot_allocator(
    *,
    mamba_end,
    take: Callable[[int], Optional[torch.Tensor]],
    state: StateStorage,
    device,
) -> UnifiedMambaSlotAllocator:
    """Install Mamba slot allocation and host capacities."""
    state_pool = state.pool
    slot_allocator = UnifiedMambaSlotAllocator(
        mamba_end,
        device=device,
        take=take,
    )
    state.slots = slot_allocator
    state_pool.host_capacity_tokens = state_pool.size
    state_pool.host_capacity_bytes = (
        state_pool.host_capacity_tokens * mamba_end.entry_bytes
    )
    return slot_allocator
