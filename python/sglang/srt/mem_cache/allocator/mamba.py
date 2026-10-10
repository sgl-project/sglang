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

from typing import Iterator, Optional

import torch


class MambaSlotAllocator:
    """Free-list of Mamba pool slot indices. Deliberately not a subclass of
    ``BaseTokenToKVPoolAllocator``: slots are per request, not per token."""

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        self._pending_free_count: int | None = None
        self._reused_pending_pages = False
        # Set by alloc_group_begin(); alloc(1) drains it until alloc_group_end().
        self._alloc_iter: Optional[Iterator] = None
        self.clear()

    def available_size(self) -> int:
        return len(self.free_slots)

    def reset_reuse_tracking(self) -> None:
        """Use the same restore interval as BaseTokenToKVPoolAllocator.reset_reuse_tracking."""
        self._pending_free_count = 0
        self._reused_pending_pages = False

    @property
    def needs_restore_fence(self) -> bool:
        return self._reused_pending_pages

    def schedulable_available_size(self) -> int:
        """Planner-facing free count. Same as ``available_size`` for a static pool;
        byte-coordinated allocators return their byte-limited view instead."""
        return self.available_size()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch of slots for match_prefix to amortize overhead."""
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._do_alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self.free(torch.cat(remaining))
        self._alloc_iter = None

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._do_alloc(need_size)

    def _do_alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None
        if self._pending_free_count:
            ready = len(self.free_slots) - self._pending_free_count
            pending = max(0, need_size - ready)
            self._reused_pending_pages |= pending > 0
            self._pending_free_count -= pending
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self._pending_free_count is not None:
            self._pending_free_count += len(free_index)
        self.free_slots = torch.cat((self.free_slots, free_index))

    def clear(self):
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        if self._pending_free_count is not None:
            self.reset_reuse_tracking()
