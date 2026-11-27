"""
Copyright 2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Slot allocator for the Mamba state pool.

Mamba caches one whole state tensor per request, so the allocator hands out
fixed-size slots (1 per request) rather than paged token KV indices.  The
underlying tensor storage lives in ``MambaPool``; this class owns only the
free-slot bookkeeping.
"""

from __future__ import annotations

from typing import Optional

import torch


class MambaSlotAllocator:
    """Free-list of Mamba pool slot indices. Deliberately not a subclass of
    ``BaseTokenToKVPoolAllocator``: slots are per request, not per token."""

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        # Set by alloc_group_begin(); alloc(1) drains it until alloc_group_end().
        self._alloc_reserve: Optional[torch.Tensor] = None
        self.clear()

    def translate(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """Static slot ids already index physical storage."""
        return slot_ids

    def available_size(self) -> int:
        return len(self.free_slots)

    def schedulable_available_size(self) -> int:
        """Planner-facing free count. Same as ``available_size`` for a static pool;
        byte-coordinated allocators return their byte-limited view instead."""
        return self.available_size()

    def slots_fit(self, count: int) -> bool:
        """Whether ``count`` slots can be taken now."""
        return self.schedulable_available_size() >= count

    def available_after_alloc_group(self) -> int:
        reserve = self._alloc_reserve
        return self.available_size() + (0 if reserve is None else len(reserve))

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch of slots for match_prefix to amortize overhead."""
        self._alloc_reserve = self._do_alloc(num_reqs) if num_reqs > 0 else None

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        reserve, self._alloc_reserve = self._alloc_reserve, None
        if reserve is not None:
            self.free(reserve)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        reserve = self._alloc_reserve
        if need_size == 1 and reserve is not None and len(reserve):
            self._alloc_reserve = reserve[1:]
            return reserve[:1]
        return self._do_alloc(need_size)

    def _do_alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor, *, owned: bool = False):
        # ``owned`` matters only to a deferred free; this one frees at once.
        if free_index.numel() == 0:
            return
        self.free_slots = torch.cat((self.free_slots, free_index))

    def clear(self):
        self._alloc_reserve = None
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
