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

from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import torch


@dataclass(frozen=True)
class MambaStateIndexInvariant:
    """Allocator guarantee consumed by the post-translation metadata producer."""

    padding_index: int = -1
    first_active_index: int = 1
    live_allocations_are_unique: bool = True
    allocated_indices_are_in_bounds: bool = True
    unified_translation_is_physical: bool = True


MAMBA_STATE_INDEX_INVARIANT = MambaStateIndexInvariant()


@dataclass(frozen=True)
class MambaStateIndexReplayProvenance:
    """Promise made by a graph runner that refreshes a stable index buffer."""

    refreshes_before_graph_launch: bool = True
    carries_exact_active_request_ids: bool = True


MAMBA_STATE_INDEX_REPLAY_PROVENANCE = MambaStateIndexReplayProvenance()


@dataclass(frozen=True)
class MambaStateIndexContract:
    """Frozen allocator attestation for one kernel-facing index buffer.

    Within ``index_tensor[:active_prefix]`` IDs are physical, unique, and in
    ``[first_active_index, state_slots)``.  The remaining replay rows are the
    ``padding_index`` sentinel.  The metadata producer issues this only after
    request gather, unified v2p translation, and padding.  Its distinct host
    request identities rule out repeated/top-k gather rows; combined with the
    allocator's unique live-slot invariant, that proves the active values
    without reading the CUDA tensor.  Tests cover both slot allocators and
    repeated stable-buffer refreshes.
    """

    version: int = 1
    padding_index: int = -1
    index_tensor: torch.Tensor | None = field(repr=False, compare=False, default=None)
    index_data_ptr: int = 0
    index_storage_offset: int = 0
    index_numel: int = 0
    active_prefix: int = 0
    state_slots: int = 0
    first_active_index: int = 1
    active_indices_are_unique: bool = True
    indices_are_physical: bool = True
    padding_is_trailing: bool = True
    source_invariant: MambaStateIndexInvariant | None = field(
        repr=False,
        compare=False,
        default=None,
    )
    active_request_ids: tuple[str, ...] = field(
        repr=False,
        compare=False,
        default=(),
    )
    replay_provenance: MambaStateIndexReplayProvenance | None = field(
        repr=False,
        compare=False,
        default=None,
    )
    allocator_id: int = 0
    _issuer: object = field(repr=False, compare=False, default=None)

    def matches(
        self,
        index_tensor: torch.Tensor,
        *,
        batch_size: int,
        state_slots: int,
    ) -> bool:
        """Check the exact tensor/envelope binding without reading its values."""
        return (
            self._issuer is _MAMBA_STATE_INDEX_CONTRACT_ISSUER
            and self.index_tensor is index_tensor
            and self.index_data_ptr == index_tensor.data_ptr()
            and self.index_storage_offset == index_tensor.storage_offset()
            and self.index_numel == index_tensor.numel() == batch_size
            and 0 <= self.active_prefix <= batch_size
            and self.state_slots == state_slots
            and self.padding_index == -1
            and self.first_active_index == 1
            and self.active_indices_are_unique
            and self.indices_are_physical
            and self.padding_is_trailing
            and self.source_invariant is MAMBA_STATE_INDEX_INVARIANT
            and len(self.active_request_ids) == self.active_prefix
            and len(set(self.active_request_ids)) == self.active_prefix
            and (
                self.replay_provenance is None
                or self.replay_provenance is MAMBA_STATE_INDEX_REPLAY_PROVENANCE
            )
            and self.allocator_id != 0
        )


_MAMBA_STATE_INDEX_CONTRACT_ISSUER = object()


def _issue_state_index_contract(
    allocator,
    index_tensor: torch.Tensor,
    *,
    active_prefix: int,
    state_slots: int,
    active_request_ids: Sequence[str],
    replay_provenance: MambaStateIndexReplayProvenance | None = None,
) -> MambaStateIndexContract:
    """Issue provenance at the post-gather/post-v2p metadata producer."""
    if allocator.state_index_invariant is not MAMBA_STATE_INDEX_INVARIANT:
        raise ValueError("Mamba allocator does not expose the frozen index invariant")
    if state_slots != allocator.size + 1:
        raise ValueError(
            "Mamba state-index contract requires allocator.size + 1 state slots: "
            f"got {state_slots=} for allocator size {allocator.size}"
        )
    if not 0 <= active_prefix <= index_tensor.numel():
        raise ValueError(
            "Mamba state-index active prefix is outside the index buffer: "
            f"got {active_prefix=} for {index_tensor.numel()=}"
        )
    request_ids = tuple(active_request_ids)
    if len(request_ids) != active_prefix:
        raise ValueError(
            "Mamba state-index contract requires one host request identity per "
            f"active row: got {len(request_ids)=} for {active_prefix=}"
        )
    if len(set(request_ids)) != active_prefix:
        raise ValueError(
            "Mamba state-index contract requires distinct active request identities"
        )
    if replay_provenance is not None and (
        replay_provenance is not MAMBA_STATE_INDEX_REPLAY_PROVENANCE
    ):
        raise ValueError("Unknown Mamba state-index replay producer")
    return MambaStateIndexContract(
        index_tensor=index_tensor,
        index_data_ptr=index_tensor.data_ptr(),
        index_storage_offset=index_tensor.storage_offset(),
        index_numel=index_tensor.numel(),
        active_prefix=active_prefix,
        state_slots=state_slots,
        source_invariant=allocator.state_index_invariant,
        active_request_ids=request_ids,
        replay_provenance=replay_provenance,
        allocator_id=id(allocator),
        _issuer=_MAMBA_STATE_INDEX_CONTRACT_ISSUER,
    )


class MambaSlotAllocator:
    """Manages the free-list of Mamba pool slot indices.

    Unlike ``BaseTokenToKVPoolAllocator`` which is designed for per-token KV
    pages, Mamba slots are request-level (typically 1 slot per request).
    We keep the interface minimal and do NOT inherit the KV base class.
    """

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        self.state_index_invariant = MAMBA_STATE_INDEX_INVARIANT
        # Active preallocated batch for `alloc_group_begin` / `alloc_group_end`.
        # When non-None, `alloc(1)` consumes the next slot from this iterator
        # instead of calling `_do_alloc(1)` per request. Reset to None outside
        # a group window so `alloc` falls through to the per-call path.
        self._alloc_iter: Optional[Iterator] = None
        self.clear()

    def available_size(self) -> int:
        return len(self.free_slots)

    def schedulable_available_size(self) -> int:
        """Planner-facing free count. Identity to ``available_size`` for the
        static pool (slot-count and byte-coordinated views coincide); the shared
        ``UnifiedMambaSlotAllocator`` overrides it with the byte-coordinated view.
        Lets ``alloc_req_slots`` call it uniformly without a getattr fallback."""
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
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.free_slots = torch.cat((self.free_slots, free_index))

    def clear(self):
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
