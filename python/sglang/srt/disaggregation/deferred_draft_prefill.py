from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class DeferredDraftPrefillPool:
    """Bounded, request-owned staging for target hidden states sent from P to D.

    These rows are separate from target KV: a decode request may keep its KV for
    thousands of tokens after its draft prefill has finished. Hidden staging is
    held only through transfer and draft initialization. Non-final prefill PP
    ranks use the same allocator without tensor storage so admission decisions
    and the state-component ordering remain uniform across pipeline ranks.

    The caller must not release an in-flight transfer's source or destination
    before the transport has reached a terminal state. CUDA release is further
    fenced against consumers still running on the scheduler stream.
    """

    def __init__(
        self,
        num_slots: int,
        max_input_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: str | torch.device,
        *,
        allocate_storage: bool = True,
    ):
        for name, value in (
            ("num_slots", num_slots),
            ("max_input_tokens", max_input_tokens),
            ("hidden_size", hidden_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if num_slots * max_input_tokens > 2**31 - 1:
            raise ValueError("Deferred draft hidden row indices must fit int32")
        self.num_slots = num_slots
        self.max_input_tokens = max_input_tokens
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.allocate_storage = allocate_storage
        self.buffer = None
        self._free_slots = deque(range(num_slots))
        self._slots: dict[str, int] = {}
        self._owners: dict[str, object] = {}
        self._lengths: dict[str, int] = {}
        self._written: dict[str, int] = {}
        self._pending: list[tuple[int, Any]] = []
        self._quarantined_slots: dict[int, str] = {}
        if allocate_storage:
            import torch

            self.buffer = torch.empty(
                (num_slots, max_input_tokens, hidden_size), dtype=dtype, device=device
            )

    @staticmethod
    def _key(req) -> str:
        rid = req if isinstance(req, str) else req.rid
        if not isinstance(rid, str) or not rid:
            raise ValueError("Deferred draft staging requires a nonempty request RID")
        return rid

    def _request_length(self, req) -> int:
        if isinstance(req, str):
            raise ValueError(
                "Initial allocation requires the request's input token IDs"
            )
        length = len(req.origin_input_ids)
        if not 0 < length <= self.max_input_tokens:
            raise ValueError(
                f"Request {self._key(req)} has {length} input tokens; deferred draft "
                f"prefill supports 1..{self.max_input_tokens}. Increase "
                "--disaggregation-draft-prefill-max-input-tokens."
            )
        return length

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

    @property
    def quarantined_slots(self) -> int:
        return len(self._quarantined_slots)

    def get_slot_stats(self) -> dict[str, int]:
        """Read ownership without reclaiming or querying CUDA events.

        ``pending_cuda`` includes completed events not yet reclaimed by normal
        allocation. It does not measure work still executing on the GPU.
        """
        return {
            "total": self.num_slots,
            "free": len(self._free_slots),
            "active": len(self._slots),
            "pending_cuda": len(self._pending),
            "quarantined": len(self._quarantined_slots),
        }

    def has_staged_rid(self, rid: str) -> bool:
        return self._key(rid) in self._slots

    def owns(self, req, slot: int | None = None) -> bool:
        key = self._key(req)
        return (
            key in self._slots
            and (isinstance(req, str) or self._owners[key] is req)
            and (slot is None or self._slots[key] == slot)
        )

    def try_alloc(self, req) -> bool:
        key = self._key(req)
        length = self._request_length(req)
        if key in self._slots:
            if not self.owns(req):
                raise ValueError(
                    f"Another active request already owns staged RID {key}"
                )
            if self._lengths[key] != length:
                raise ValueError(f"Input length changed for staged request {key}")
            return True
        self.reclaim()
        if not self._free_slots:
            return False
        self._slots[key] = self._free_slots.popleft()
        self._owners[key] = req
        self._lengths[key] = length
        self._written[key] = 0
        return True

    def alloc(self, req) -> int:
        if not self.try_alloc(req):
            raise RuntimeError("Deferred draft prefill staging is full")
        return self.get_slot(req)

    def get_slot(self, req) -> int:
        if not self.owns(req):
            raise KeyError(f"Request does not own hidden staging: {self._key(req)}")
        return self._slots[self._key(req)]

    def write_chunk(self, req, hidden: torch.Tensor, start: int, end: int) -> None:
        key = self._key(req)
        slot = self.get_slot(req)
        if self.buffer is None:
            raise RuntimeError("This PP rank does not own hidden staging storage")
        if not (start == self._written[key] and start < end <= self._lengths[key]):
            raise ValueError(
                f"Noncontiguous target hidden chunk for {key}: [{start}, {end}), "
                f"expected start {self._written[key]} within {self._lengths[key]}"
            )
        if tuple(hidden.shape) != (end - start, self.hidden_size):
            raise ValueError(
                f"Target hidden chunk shape {tuple(hidden.shape)} does not match "
                f"({end - start}, {self.hidden_size})"
            )
        if hidden.dtype != self.buffer.dtype or hidden.device != self.buffer.device:
            raise ValueError("Target hidden dtype/device differs from staging storage")
        self.buffer[slot, start:end].copy_(hidden)
        self._written[key] = end

    def mark_received(self, req, length: int) -> None:
        key = self._key(req)
        self.get_slot(req)
        if length != self._lengths[key]:
            raise ValueError(
                f"Received hidden length {length} differs from input length "
                f"{self._lengths[key]} for {key}"
            )
        self._written[key] = length

    def get(self, req, length: int | None = None) -> torch.Tensor:
        key = self._key(req)
        slot = self.get_slot(req)
        length = self._lengths[key] if length is None else length
        if self.buffer is None:
            raise RuntimeError("This PP rank does not own hidden staging storage")
        if length != self._lengths[key] or self._written[key] != length:
            raise RuntimeError(f"Target hidden prefix is incomplete for {key}")
        return self.buffer[slot, :length]

    def transfer_indices(
        self, req, length: int | None = None, *, start: int = 0
    ) -> list[int]:
        key = self._key(req)
        slot = self.get_slot(req)
        length = self._lengths[key] if length is None else length
        if not (0 <= start <= length <= self._lengths[key]):
            raise ValueError(
                f"Invalid target hidden transfer range [{start}, {length})"
            )
        if not self.allocate_storage:
            return []
        first = slot * self.max_input_tokens
        return list(range(first + start, first + length))

    def free(self, req, completion_event=None, *, fence_consumers: bool = True) -> bool:
        key = self._key(req)
        if not self.owns(req):
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
        slot = self._slots.pop(key)
        del self._owners[key]
        del self._lengths[key]
        del self._written[key]
        if completion_event is None:
            self._free_slots.append(slot)
        else:
            self._pending.append((slot, completion_event))
        return True

    def quarantine(self, req, *, reason: str) -> bool:
        """Retain storage permanently when a peer operation may still use it.

        Remove the RID mapping so a later request with the same RID gets fresh
        storage. A tombstoned slot never enters either the free or event queue;
        a local CUDA event cannot prove that a remote RDMA operation has ended.
        """
        key = self._key(req)
        if not self.owns(req):
            return False
        slot = self._slots.pop(key)
        del self._owners[key]
        del self._lengths[key]
        del self._written[key]
        self._quarantined_slots[slot] = reason
        logger.warning(
            "Quarantined deferred draft-prefill hidden slot %d for request %s: %s",
            slot,
            key,
            reason,
        )
        return True

    def free_source(self, req) -> bool:
        """Release after source transfer drain and the existing PP RID consensus.

        A sender's terminal/drained transfer has already finished reading these
        rows. It has no draft consumer to fence. Avoid creating a fresh event
        whose query result could make admission diverge between PP ranks.
        """
        return self.free(req, fence_consumers=False)

    def get_contiguous_buf_infos(self) -> tuple[list[int], list[int], list[int]]:
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
    src_ptrs: list[int],
    src_item_lens: list[int],
    dst_ptrs: list[int],
    dst_item_lens: list[int],
    src_indices: list[int],
    dst_indices: list[int],
) -> list[tuple[int, int, int]]:
    """Map token rows without KV head/PP-layer slicing or uninitialized padding.

    Target hidden states are replicated on all D attention-TP ranks. The final
    P pipeline rank owns exactly one tensor; non-final ranks publish an empty
    component in the same wire position and send no rows.
    """
    if not src_ptrs and not src_item_lens and not src_indices:
        return []
    if not (
        len(src_ptrs) == len(src_item_lens) == len(dst_ptrs) == len(dst_item_lens) == 1
    ):
        raise ValueError("Deferred draft hidden transfer needs one tensor on each peer")
    if src_ptrs[0] <= 0 or dst_ptrs[0] <= 0:
        raise ValueError("Deferred draft hidden transfer received an invalid pointer")
    if src_item_lens[0] <= 0 or src_item_lens != dst_item_lens:
        raise ValueError("Deferred draft hidden dtype/width differs between peers")
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
