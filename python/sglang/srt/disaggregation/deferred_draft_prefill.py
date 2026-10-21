"""Bounded target-hidden staging and safe ownership for deferred draft prefill."""

from __future__ import annotations

import enum
import logging
import threading
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class DeferredDraftPrefillPool:
    """Keep prompt hidden rows until transfer or local draft consumption ends.

    Requests are keyed by object identity, so a later request reusing a RID
    cannot release an earlier request's storage. Non-final P ranks keep the
    same bookkeeping without allocating a tensor, preserving PP admission.
    """

    def __init__(
        self,
        num_slots,
        max_input_tokens,
        hidden_size,
        dtype,
        device,
        *,
        allocate_storage=True,
    ):
        if min(num_slots, max_input_tokens, hidden_size) <= 0:
            raise ValueError("Deferred draft staging dimensions must be positive")
        if num_slots * max_input_tokens > 2**31 - 1:
            raise ValueError("Deferred draft hidden row indices must fit int32")
        self.num_slots = num_slots
        self.max_input_tokens = max_input_tokens
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.allocate_storage = allocate_storage
        self._free_slots = deque(range(num_slots))
        # Req uses object identity/hash; retaining the key also retains ownership.
        self._slots: dict[object, tuple[int, int, int]] = {}
        self._pending = []
        self.buffer = None
        if allocate_storage:
            import torch

            self.buffer = torch.empty(
                (num_slots, max_input_tokens, hidden_size), dtype=dtype, device=device
            )

    def reclaim(self) -> None:
        pending = []
        for slot, event in self._pending:
            if event.query():
                self._free_slots.append(slot)
            else:
                pending.append((slot, event))
        self._pending = pending

    @property
    def available_slots(self) -> int:
        self.reclaim()
        return len(self._free_slots)

    def owns(self, req, slot=None) -> bool:
        return req in self._slots and (slot is None or self._slots[req][0] == slot)

    def try_alloc(self, req) -> bool:
        length = len(req.origin_input_ids)
        if not 0 < length <= self.max_input_tokens:
            raise ValueError(
                f"Deferred draft prefill requires 1..{self.max_input_tokens} input tokens"
            )
        if req in self._slots:
            if self._slots[req][1] != length:
                raise ValueError("Input length changed for a staged request")
            return True
        self.reclaim()
        if not self._free_slots:
            return False
        self._slots[req] = (self._free_slots.popleft(), length, 0)
        return True

    def alloc(self, req) -> int:
        if not self.try_alloc(req):
            raise RuntimeError("Deferred draft prefill staging is full")
        return self.get_slot(req)

    def get_slot(self, req) -> int:
        return self._slots[req][0]

    def write_chunk(self, req, hidden: torch.Tensor, start: int, end: int) -> None:
        slot, length, written = self._slots[req]
        if self.buffer is None or not (start == written and start < end <= length):
            raise ValueError("Target hidden chunks must cover the prompt in order")
        if tuple(hidden.shape) != (end - start, self.hidden_size):
            raise ValueError("Target hidden chunk has an unexpected shape")
        if hidden.dtype != self.buffer.dtype or hidden.device != self.buffer.device:
            raise ValueError("Target hidden dtype/device differs from staging storage")
        self.buffer[slot, start:end].copy_(hidden)
        self._slots[req] = (slot, length, end)

    def mark_received(self, req, length: int) -> None:
        slot, expected, _ = self._slots[req]
        if length != expected:
            raise ValueError("Received hidden length differs from the prompt")
        self._slots[req] = (slot, length, length)

    def get(self, req, length=None) -> torch.Tensor:
        slot, expected, written = self._slots[req]
        length = expected if length is None else length
        if self.buffer is None or length != expected or written != length:
            raise RuntimeError("Target hidden prefix is incomplete")
        return self.buffer[slot, :length]

    def transfer_indices(self, req, length=None, *, start=0) -> list[int]:
        slot, expected, _ = self._slots[req]
        length = expected if length is None else length
        if not 0 <= start <= length <= expected:
            raise ValueError("Invalid target hidden transfer range")
        if self.buffer is None:
            return []
        first = slot * self.max_input_tokens
        return list(range(first + start, first + length))

    def free(self, req, completion_event=None, *, fence_consumers=True) -> bool:
        if req not in self._slots:
            return False
        if (
            fence_consumers
            and completion_event is None
            and self.buffer is not None
            and self.buffer.is_cuda
        ):
            import torch

            completion_event = torch.cuda.Event()
            completion_event.record(torch.cuda.current_stream(self.buffer.device))
        slot, _, _ = self._slots.pop(req)
        if completion_event is None:
            self._free_slots.append(slot)
        else:
            self._pending.append((slot, completion_event))
        return True

    def free_source(self, req) -> bool:
        # RDMA has drained; a fresh CUDA event could diverge PP admission.
        return self.free(req, fence_consumers=False)

    def quarantine(self, req, *, reason: str) -> bool:
        if req not in self._slots:
            return False
        slot, _, _ = self._slots.pop(req)
        # Never return an uncertain RDMA destination/source to either free queue.
        logger.warning(
            "Quarantined deferred draft hidden slot %d for %s: %s",
            slot,
            req.rid,
            reason,
        )
        return True

    def get_contiguous_buf_infos(self):
        if self.buffer is None:
            return [], [], []
        return (
            [self.buffer.data_ptr()],
            [self.buffer.numel() * self.buffer.element_size()],
            [self.hidden_size * self.buffer.element_size()],
        )

    @property
    def wire_layout(self) -> dict:
        return {"version": 1, "hidden_size": self.hidden_size, "dtype": str(self.dtype)}


def build_deferred_draft_hidden_transfer_blocks(
    src_ptrs, src_item_lens, dst_ptrs, dst_item_lens, src_indices, dst_indices
):
    """Replicate contiguous hidden rows, without KV head or PP-layer slicing."""
    if not src_ptrs and not src_item_lens and not src_indices:
        return []
    if not (
        len(src_ptrs) == len(src_item_lens) == len(dst_ptrs) == len(dst_item_lens) == 1
    ):
        raise ValueError("Deferred draft hidden transfer needs one tensor on each peer")
    if (
        min(src_ptrs[0], dst_ptrs[0], src_item_lens[0]) <= 0
        or src_item_lens != dst_item_lens
    ):
        raise ValueError(
            "Deferred draft hidden pointer or dtype/width differs between peers"
        )
    if not src_indices or len(src_indices) != len(dst_indices):
        raise ValueError("Deferred draft hidden token counts differ between peers")
    for indices in (src_indices, dst_indices):
        if indices[0] < 0 or any(b != a + 1 for a, b in zip(indices, indices[1:])):
            raise ValueError(
                "Deferred draft hidden rows must be contiguous and nonnegative"
            )
    item_len = src_item_lens[0]
    return [
        (
            src_ptrs[0] + src_indices[0] * item_len,
            dst_ptrs[0] + dst_indices[0] * item_len,
            len(src_indices) * item_len,
        )
    ]


class DeferredDraftPrefillRetirementState(str, enum.Enum):
    FREE = "free"
    HOLD = "hold"
    QUARANTINE = "quarantine"


class DeferredDraftPrefillSourceFence:
    """One final chunk's ownership, shared by its sender and transfer worker."""

    def __init__(self):
        self._lock = threading.Lock()
        self._submitted = self._sealed = self._settled = False
        self.poison_reason = None

    def _state(self):
        if self.poison_reason is not None:
            return DeferredDraftPrefillRetirementState.QUARANTINE
        if self._sealed and (not self._submitted or self._settled):
            return DeferredDraftPrefillRetirementState.FREE
        return DeferredDraftPrefillRetirementState.HOLD

    @property
    def retirement_state(self):
        with self._lock:
            return self._state()

    def submit(self) -> bool:
        # Called before queue.put, including chunks not yet counted by a worker.
        with self._lock:
            if self._sealed or self._submitted or self.poison_reason is not None:
                return False
            self._submitted = True
            return True

    def seal(self):
        with self._lock:
            self._sealed = True
            return self._state()

    def settled(self):
        with self._lock:
            self._settled = True
            return self._state()

    def poison(self, reason="Transfer drain could not be proven"):
        with self._lock:
            if self.poison_reason is None:
                self.poison_reason = reason
            return self._state()


def deferred_draft_prefill_retirement_key(req):
    generation = req.pd_deferred_draft_source_generation
    if not generation:
        raise ValueError("Deferred draft source retirement needs an intake generation")
    return req.rid, req.bootstrap_room, generation


def merge_deferred_draft_prefill_retirements(left, right):
    """Intersect the same generations; an unsafe stage vetoes reclamation."""
    state = DeferredDraftPrefillRetirementState
    merged = {}
    for key in left.keys() & right.keys():
        votes = (left[key], right[key])
        merged[key] = (
            state.QUARANTINE
            if state.QUARANTINE in votes
            else (state.HOLD if state.HOLD in votes else state.FREE)
        )
    return merged


def _source_fence(req):
    return getattr(
        getattr(req, "disagg_kv_sender", None), "deferred_draft_prefill_fence", None
    )


class DeferredDraftPrefillSourceRetirements:
    """Retire metadata and source hidden together after a full PP consensus."""

    def __init__(self, allocator):
        self.allocator = allocator
        self._entries = {}

    def __len__(self):
        return len(self._entries)

    def register(self, req) -> bool:
        pool = req.deferred_draft_prefill_pool
        if pool is None or not pool.owns(req):
            return False
        key = deferred_draft_prefill_retirement_key(req)
        if key in self._entries:
            if self._entries[key] is not req:
                raise RuntimeError("Deferred draft source generation was reused")
            return True
        if req.metadata_buffer_index < 0:
            raise RuntimeError("Hidden source retirement lost its metadata ownership")
        fence = _source_fence(req)
        if fence is not None:
            fence.seal()
        self._entries[key] = req
        return True

    def snapshot(self):
        return {
            key: fence.retirement_state
            if (fence := _source_fence(req)) is not None
            else DeferredDraftPrefillRetirementState.QUARANTINE
            for key, req in self._entries.items()
        }

    def resolve(self, states=None):
        state = DeferredDraftPrefillRetirementState
        for key, vote in (self.snapshot() if states is None else states).items():
            req = self._entries.get(key)
            if req is None or vote == state.HOLD:
                continue
            pool, fence = req.deferred_draft_prefill_pool, _source_fence(req)
            if vote == state.FREE:
                if fence is None or fence.retirement_state != state.FREE:
                    raise RuntimeError("Unsafe deferred draft source release")
                pool.free_source(req)
                self.allocator.free(req.metadata_buffer_index)
            else:
                pool.quarantine(
                    req,
                    reason=(fence.poison_reason if fence is not None else None)
                    or "A pipeline peer could not prove its source transfer drained",
                )
            req.metadata_buffer_index = -1
            del self._entries[key]


def validate_deferred_draft_prefill_request(req, tokenized_req, max_input_tokens):
    if not 0 < len(req.origin_input_ids) <= max_input_tokens:
        return f"Deferred draft prefill requires 1..{max_input_tokens} input tokens"
    for field in (
        "mm_inputs",
        "input_embeds",
        "session_params",
        "session_id",
        "lora_id",
        "custom_logit_processor",
    ):
        if getattr(tokenized_req, field, None) is not None:
            return "Deferred draft prefill accepts plain text requests without sessions or custom processors"
    if any(
        getattr(req.sampling_params, field, None) is not None
        for field in ("json_schema", "regex", "ebnf", "structural_tag")
    ):
        return "Deferred draft prefill does not support constrained decoding"
    return None
