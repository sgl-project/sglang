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

from typing import TYPE_CHECKING, Iterator, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MambaPool


class MambaTokenToKVPoolAllocator:
    """Manages the free-list of Mamba pool slot indices.

    Unlike ``BaseTokenToKVPoolAllocator`` which is designed for per-token KV
    pages, Mamba slots are request-level (typically 1 slot per request).
    We keep the interface minimal and do NOT inherit the KV base class.
    """

    def __init__(self, size: int, device: str, mamba_pool: "MambaPool"):
        self.size = size
        self.device = device
        self._mamba_pool = mamba_pool

        self._is_not_in_free_group = True
        self._free_group: list[torch.Tensor] = []
        self._alloc_iter: Optional[Iterator] = None

        self.clear()

    @property
    def mamba_pool(self) -> "MambaPool":
        return self._mamba_pool

    def available_size(self) -> int:
        return len(self.free_slots)

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
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self._is_not_in_free_group:
            self.free_slots = torch.cat((self.free_slots, free_index))
        else:
            self._free_group.append(free_index)

    def free_group_begin(self):
        self._is_not_in_free_group = False
        self._free_group = []

    def free_group_end(self):
        self._is_not_in_free_group = True
        if self._free_group:
            self.free(torch.cat(self._free_group))

    def clear(self):
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self._is_not_in_free_group = True
        self._free_group = []

    def clear_slots(self, indices: torch.Tensor):
        """Zero out mamba state at the given pool indices (delegates to pool)."""
        self._mamba_pool.clear_slots(indices)

    def backup_state(self) -> torch.Tensor:
        return self.free_slots.clone()

    def restore_state(self, state: torch.Tensor):
        self.free_slots = state
