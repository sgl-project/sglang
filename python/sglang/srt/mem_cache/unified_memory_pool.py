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
"""The unified placement's per-layout views over a `KVArena`, and the slot
allocator of its state band.

Every view below is a layout's per-layer tensors over one band of the arena;
the arena owns the bytes and moves and zeroes their units. Everything above
the allocator stores virtual ids; the allocator owns the per-sub-pool
virtual<->physical tables, and compaction only mutates those (no reference
rewriting).
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.kv.memory.arena import GB, KVArena
from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)


class UnifiedMambaPool(MambaPool):
    """Mamba state pool whose conv/temporal state are strided views into a
    `KVArena` band.

    Pure PHYSICAL store: slots and their v<->p mapping are the chain allocator's
    state band, and the arena moves them. Does NOT call `super().__init__()` --
    replicates the minimal `MambaPool` state against the arena so inherited
    methods work.
    """

    def __init__(
        self,
        *,
        arena: KVArena,
        spec: MambaSubPoolSpec,
        spec_state_size: int,
        mamba_layer_ids: List[int],
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        assert spec.layer_num == len(mamba_layer_ids)
        # PP disagg state transfer maps entries by global layer id.
        self.mamba_layer_ids = list(mamba_layer_ids)
        conv_views, temporal_view = spec.views(arena)
        max_slots = arena.extent_entries(spec.name)

        self._arena = arena
        self._sub_pool_name = spec.name

        # Replicate the state MambaPool.__init__ would have set.
        self._max_size = max_slots - 1  # -1 for reserved slot 0
        self.size = self._max_size
        self.device = arena.device
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.num_mamba_layers = spec.layer_num
        # GDN/KDA ReplaySSM / spec unsupported; replicate parent's disabled-state
        # attrs so unconditional reads (e.g. `replayssm_cache_base is not None` in
        # the req-slot alloc path) and `... is not None` guards don't AttributeError.
        self.enable_linear_replayssm = False
        self.linear_replayssm_cache_len = 16
        self.replayssm_write_pos = None
        self.replayssm_is_kda = False
        self.enable_linear_replayssm_spec = False
        self.replayssm_spec_fold = False
        self.replayssm_spec_write_pos = None
        self.replayssm_cache_base = None
        self.replayssm_is_flush = None
        self.debug_memory_pool = False
        self.conv_shard_groups = None
        self.conv_slice_axis = spec.conv_slice_axis

        assert conv_views[0].shape[0] == self.num_mamba_layers, (
            f"conv_views layers={conv_views[0].shape[0]} vs expected {self.num_mamba_layers}"
        )
        assert conv_views[0].shape[1] == self._max_size + 1, (
            f"conv_views slots={conv_views[0].shape[1]} vs expected {self._max_size + 1}"
        )

        # Per-draft-token intermediate buffers have a different outer size
        # (spec_state_size+1), so they're NOT in the shared buffer; allocate locally.
        temporal_state_shape = spec.temporal_state_shape
        conv_state_shape = spec.conv_state_shapes
        conv_dtype = spec.conv_dtype
        ssm_dtype = spec.temporal_dtype
        if speculative_num_draft_tokens is not None:
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        self.num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device=arena.device,
                )
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            self.num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            cshape[0],
                            cshape[1],
                        ),
                        dtype=conv_dtype,
                        device=arena.device,
                    )
                    for cshape in conv_state_shape
                ]
            self.mamba_cache = self.SpeculativeState(
                conv=list(conv_views),
                temporal=temporal_view,
                intermediate_ssm=intermediate_ssm_state_cache,
                intermediate_conv_window=intermediate_conv_window_cache,
            )
        else:
            self.mamba_cache = self.State(conv=list(conv_views), temporal=temporal_view)

        self.mem_usage = arena.total_bytes / GB
        logger.info(
            "[unified-memory-pool] UnifiedMambaPool(%s) over the KV arena: max_slots=%d, "
            "num_mamba_layers=%d",
            spec.name,
            max_slots,
            self.num_mamba_layers,
        )
        self._entry_bytes = spec.entry_bytes()

    def register_slot_state(self, state) -> None:
        # The arena moves whole entries, so every per-slot state must live in
        # the entry: side states registered beside it would be left behind.
        raise ValueError(
            "--enable-unified-memory keeps a slot's whole state in its entry; "
            "side states registered beside the state pool are not supported"
        )

    # State ops take PHYSICAL slot ids and act on whole entries, so a fused
    # draft state block is cleared and copied with the host streams.

    def clear_slots(self, indices: torch.Tensor):
        # index_fill_, not `view[indices] = 0`: assigning a Python scalar
        # through an index tensor synchronizes the host.
        self._arena.entries(self._sub_pool_name).index_fill_(
            0, indices.to(torch.int64), 0
        )

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
        assert self.replayssm_write_pos is None
        entries = self._arena.entries(self._sub_pool_name)
        entries[dst_indices] = entries[src_indices]

    # -- PD state transfer (StateType.MAMBA) --
    # The transfer item is the whole per-slot envelope, addressed as
    # `raw_ptr + physical_slot * entry_bytes`. An envelope cannot be TP-resliced
    # or PP-subset, so the per-tensor metadata below stays empty and both sides
    # must build identical mamba specs (equal attn TP, pp=1).

    def get_contiguous_buf_infos(self):
        raw = self._arena.raw
        return [raw.data_ptr()], [raw.numel()], [self._entry_bytes]

    def get_state_dim_per_tensor(self):
        return []

    def get_state_layer_ids(self):
        return []

    def get_state_slice_outer_counts(self):
        return []

    def get_state_conv_shard_groups(self):
        return []


class UnifiedMambaSlotAllocator:
    """Mamba slot allocator (PHYSICAL view) for the unified memory pool.

    Owns slot alloc/free, sizing, and the v<->p mapping (``translate``), presenting the
    upstream ``MambaSlotAllocator`` interface. ``alloc()`` returns VIRTUAL ids and does
    NOT clear state — clearing is deferred to ``UnifiedMambaPool.clear_slots``.
    """

    def __init__(
        self,
        mea,
        max_size: int,
        device: str,
        take: Optional[Callable[[int], Optional[torch.Tensor]]] = None,
    ):
        self._multi_ended_allocator = mea
        # Slots are taken through the chain allocator when one owns the band.
        self._take = mea.alloc if take is None else take
        self._max_size = max_size  # excludes reserved slot 0
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
        return self._max_size

    def available_size(self) -> int:
        # Slot-conservation count (max - allocated): the leak-check view, NOT the
        # planner value (use schedulable_available_size for that).
        return self._max_size - self._multi_ended_allocator.allocated_count()

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
        max_size=state_pool.size,
        device=device,
        take=take,
    )
    state.slots = slot_allocator
    state_pool.host_capacity_tokens = state_pool.size
    state_pool.host_capacity_bytes = (
        state_pool.host_capacity_tokens * mamba_end.entry_bytes
    )
    return slot_allocator
