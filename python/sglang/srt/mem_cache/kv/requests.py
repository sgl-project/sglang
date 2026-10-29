"""Which ids each request holds: one row per request, a column per sub-pool kind
of id the plan gives a request.

A row holds the owner's token ids by position (`req_to_token`) and, when the
plan has a state sub-pool, the request's state slot and, with the extra buffer,
the tracking slots it cycles through. Ids only: the bytes live in the arena, and
the allocator's id spaces translate the ids a row holds. Row 0 is padding:
CUDA-graph padded batches read and write it.

The scheduler, the attention backends and the speculative workers read the
columns under the names a static request pool gives them, so one request
table serves both placements.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional

import msgspec
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

_MAMBA_DEBUG_ASSERTS = os.environ.get("SGLANG_MAMBA_DEBUG_ASSERTS", "0") == "1"


class StateTracking(msgspec.Struct, frozen=True, kw_only=True):
    """How a request tracks its state for the prefix cache: with the extra
    buffer it holds `track_slots` tracking slots (taking `initial_slots` of
    them up front), otherwise none."""

    extra_buffer: bool
    lazy: bool
    track_slots: int
    initial_slots: int


class RequestTable:
    """The rows requests hold, and in each row the ids of the plan's sub-pools."""

    def __init__(
        self,
        *,
        size: int,
        pre_alloc_size: int,
        max_context_len: int,
        state_sub_pool: Optional[str],
        tracking: Optional[StateTracking],
        device: str,
        enable_memory_saver: bool,
        reserved_rows: Optional[int] = None,
    ):
        assert (state_sub_pool is None) == (tracking is None), (
            "a state sub-pool comes with its tracking, and only then"
        )
        # Running requests; rows for preallocated transfers come on top.
        self.size = size
        self.pre_alloc_size = pre_alloc_size
        self._num_rows = (
            reserved_rows if reserved_rows is not None else size + pre_alloc_size + 1
        )
        assert self._num_rows >= size + pre_alloc_size + 1
        self.max_context_len = max_context_len
        self.device = device
        self.state_sub_pool = state_sub_pool
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._num_rows, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(1, self._num_rows))
        # Bumped on every hand-out: a reader detects a reused row by comparing
        # the generation it stored.
        self.req_generation = torch.zeros(self._num_rows, dtype=torch.int64)

        tracking = tracking or StateTracking(
            extra_buffer=False, lazy=False, track_slots=0, initial_slots=0
        )
        self.enable_mamba_extra_buffer = tracking.extra_buffer
        self.enable_mamba_extra_buffer_lazy = tracking.lazy
        self.mamba_ping_pong_track_buffer_size = tracking.track_slots
        self.mamba_initial_tracking_slots = tracking.initial_slots
        if state_sub_pool is not None:
            self.req_index_to_mamba_index_mapping = torch.zeros(
                self._num_rows, dtype=torch.int32, device=device
            )
            if tracking.extra_buffer:
                # int64, as the slot tensors written into it.
                self.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
                    (self._num_rows, tracking.track_slots),
                    dtype=torch.int64,
                    device=device,
                )

    @property
    def holds_state_slots(self) -> bool:
        """Whether a row names its request's state slot."""
        return self.state_sub_pool is not None

    # -- rows --

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        """Bind a row to each request that holds none (a later chunk keeps its
        own); None when too few rows are free."""
        reusing = [r for r in reqs if r.kv.holds_kv]
        assert all(r.kv.kv_allocated_len > 0 for r in reusing), (
            "a reused row must carry allocated KV"
        )
        rows = self.alloc_rows(len(reqs) - len(reusing))
        if rows is None:
            return None
        fresh = iter(rows)
        for r in reqs:
            if not r.kv.holds_kv:
                r.kv.req_pool_idx = next(fresh)
        return [r.kv.req_pool_idx for r in reqs]

    def alloc_rows(self, need_size: int) -> Optional[List[int]]:
        """Take rows from the tail of the free list, with no request bound."""
        if need_size > len(self.free_slots):
            return None
        if need_size == 0:
            return []
        rows = self.free_slots[-need_size:]
        del self.free_slots[-need_size:]
        self.req_generation[rows] += 1
        return rows

    def free_rows(self, rows: List[int]) -> None:
        self.free_slots.extend(rows)

    def free(self, req: Req) -> None:
        assert req.kv.holds_kv, "request must have req_pool_idx"
        self.free_rows([req.kv.req_pool_idx])
        req.kv.req_pool_idx = None

    def write(self, indices, values) -> None:
        self.req_to_token[indices] = values

    def clear(self) -> None:
        logger.info("Reset the request table")
        self.free_slots = list(range(1, self._num_rows))
        # The generation stays monotonic across a reset.
        if self.holds_state_slots:
            self.req_index_to_mamba_index_mapping.zero_()
            if self.enable_mamba_extra_buffer:
                self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()

    # A unified table keeps no auxiliary per-row cache: the scheduler's
    # auxiliary-cache calls are identities here.

    def schedulable_token_capacity(self, physical_capacity: int) -> int:
        return physical_capacity

    def reset_aux_cache_allocator(self) -> None:
        pass

    def alloc_aux_to_lengths(
        self,
        *,
        req_pool_indices_cpu: torch.Tensor,
        target_seq_lens_cpu: torch.Tensor,
    ) -> None:
        pass

    # -- state slot columns --

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - mamba_next_track_idx
        return mamba_next_track_idx

    def get_mamba_ping_pong_keep_idx(self, req: Req) -> int:
        """The tracking slot holding the most recent tracked state."""
        return req.kv.mamba_last_track_idx

    def set_mamba_ping_pong_slot(self, req: Req, idx: int, value) -> None:
        """Set one tracking slot of the request (a slot tensor, or -1 to empty
        it) and keep its row in step: the track-save reads the row."""
        if torch.is_tensor(value):
            req.kv.mamba_ping_pong_track_buffer[idx] = value
        else:
            assert value == -1, f"expected a slot tensor or -1, got {value!r}"
            # fill_ stays on the device; assigning the int is a blocking host copy.
            req.kv.mamba_ping_pong_track_buffer[idx].fill_(-1)
        self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.kv.req_pool_idx] = (
            req.kv.mamba_ping_pong_track_buffer
        )
        req.kv.mamba_ping_pong_empty[idx] = not torch.is_tensor(value)

    def donate_mamba_ping_pong_slot(
        self, req: Req, new_slot: torch.Tensor
    ) -> torch.Tensor:
        """Hand the tracked state's slot to the prefix cache and track on in
        ``new_slot``; returns the donated slot (shape [1])."""
        donate_idx = self.get_mamba_ping_pong_keep_idx(req)
        donated = req.kv.mamba_ping_pong_track_buffer[donate_idx].unsqueeze(-1).clone()
        if _MAMBA_DEBUG_ASSERTS:
            # .item() synchronizes the stream; only pay it when debugging.
            assert donated.item() != -1, (
                f"Donated mamba slot is -1: donate_idx={donate_idx}, "
                f"buf={req.kv.mamba_ping_pong_track_buffer.tolist()}, "
                f"next_track_idx={req.kv.mamba_next_track_idx}, "
                f"rid={req.rid}"
            )
        self.set_mamba_ping_pong_slot(req, donate_idx, new_slot[0])
        return donated

    def clone_with_new_mamba(self, **kwargs):
        raise ValueError(
            "--enable-unified-memory: a draft head with its own recurrent state "
            "binds it through the target's fused draft placement"
        )
