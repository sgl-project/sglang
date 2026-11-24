"""Request-slot storage and batch metadata for trace replay."""

from dataclasses import dataclass, replace
from typing import Optional

import torch

from sglang.kernels.ops.sampling import trace_replay as trace_replay_op
from sglang.srt.utils.device import (
    async_h2d,
    get_device_module,
    is_pin_memory_available,
)


@dataclass(frozen=True)
class TraceReplaySlot:
    index: int
    generation: int
    # The tokenizer publishes a request-local list. Scheduling may cap the
    # effective length, but never mutates the published token sequence.
    token_ids: list[int]
    length: int


class TraceReplayPool:
    """Device token table addressed by stable request-pool slots.

    The first trace allocates the table. Rows are uploaded on the stream that
    consumes them, avoiding writes from the schedule stream while an overlap
    forward is reading a previous generation. A delayed sampler can restore
    its own generation from its snapshot if a newer request used the slot.
    """

    def __init__(self, req_pool):
        self.device = torch.device(req_pool.device)
        self.capacity = req_pool.req_generation.numel()
        self.max_width = req_pool.max_context_len
        self.tokens: Optional[torch.Tensor] = None
        self._residents: dict[int, tuple[int, list[int], int]] = {}
        self._stream = None
        self.revision = 0

    def _use_current_stream(self):
        if self.device.type == "cpu":
            return
        stream = get_device_module().current_stream(self.device)
        if self._stream is not None and stream != self._stream:
            stream.wait_stream(self._stream)
            if self.tokens is not None:
                self.tokens.record_stream(stream)
        self._stream = stream

    def _ensure_width(self, width: int):
        if width > self.max_width:
            raise ValueError(
                "Trace length exceeds the request pool's context capacity."
            )
        old = self.tokens
        old_width = old.shape[1] if old is not None else 0
        if width <= old_width:
            return
        # Keep slot IDs stable while growing the token dimension occasionally.
        new_width = min(self.max_width, max(width, 2 * old_width))
        self.tokens = torch.empty(
            (self.capacity, new_width), dtype=torch.int32, device=self.device
        )
        if old is not None:
            self.tokens[:, :old_width].copy_(old)
        else:
            self.tokens[0, 0] = 0

    def prepare(self, batch):
        self._use_current_stream()
        if batch._prepared_revision == self.revision:
            return self.tokens

        pending = []
        for slot in batch.slots:
            if slot is None:
                continue
            resident = self._residents.get(slot.index)
            if (
                resident is None
                or resident[0] != slot.generation
                or resident[1] is not slot.token_ids
                or resident[2] < slot.length
            ):
                pending.append(slot)

        if pending:
            self._ensure_width(max(slot.length for slot in pending))
            pin = is_pin_memory_available(self.device)
            for slot in pending:
                host = torch.tensor(
                    slot.token_ids[: slot.length], dtype=torch.int32, pin_memory=pin
                )
                self.tokens[slot.index, : slot.length].copy_(host, non_blocking=pin)
                self._residents[slot.index] = (
                    slot.generation,
                    slot.token_ids,
                    slot.length,
                )
            self.revision += 1
        batch._prepared_revision = self.revision
        return self.tokens


@dataclass
class TraceReplayBatch:
    pool: TraceReplayPool
    slots: tuple[Optional[TraceReplaySlot], ...]
    indices: torch.Tensor
    lengths: torch.Tensor
    steps: torch.Tensor
    # Only this host-side cache changes in place. In-flight forward snapshots
    # retain their slot bindings and step tensors.
    _prepared_revision: int = -1

    @classmethod
    def from_requests(cls, reqs, req_pool):
        traces = [
            getattr(req.sampling_params, "trace_decode_token_ids", None) for req in reqs
        ]
        if not any(isinstance(trace, list) and trace for trace in traces):
            return None

        slots = []
        for req, trace in zip(reqs, traces, strict=True):
            if not isinstance(trace, list) or not trace:
                slots.append(None)
                continue
            cap = req.sampling_params.max_new_tokens
            length = len(trace) if cap is None else min(len(trace), cap)
            if length <= 0:
                slots.append(None)
                continue
            index = req.kv.req_pool_idx
            assert index is not None and index > 0
            slots.append(
                TraceReplaySlot(
                    index, int(req_pool.req_generation[index]), trace, length
                )
            )

        if not any(slot is not None for slot in slots):
            return None
        pool = getattr(req_pool, "trace_replay_pool", None)
        if pool is None:
            pool = req_pool.trace_replay_pool = TraceReplayPool(req_pool)
        meta = async_h2d(
            [
                [slot.index if slot is not None else 0 for slot in slots],
                [slot.length if slot is not None else 0 for slot in slots],
                [
                    len(req.output_ids) if slot is not None else 0
                    for req, slot in zip(reqs, slots, strict=True)
                ],
            ],
            dtype=torch.int32,
            device=pool.device,
        )
        return cls(pool, tuple(slots), meta[0], meta[1], meta[2])

    def filter(self, keep_indices, keep_indices_device):
        slots = tuple(self.slots[i] for i in keep_indices)
        if not any(slot is not None for slot in slots):
            return None
        return TraceReplayBatch(
            self.pool,
            slots,
            self.indices[keep_indices_device],
            self.lengths[keep_indices_device],
            self.steps[keep_indices_device],
            _prepared_revision=self._prepared_revision,
        )

    @classmethod
    def merge(cls, left, right, left_size, right_size):
        if left is None and right is None:
            return None
        trace = left if left is not None else right
        if left is None or right is None:
            before = left_size if left is None else 0
            after = right_size if right is None else 0
            pad = (before, after)
            return cls(
                trace.pool,
                (None,) * before + trace.slots + (None,) * after,
                torch.nn.functional.pad(trace.indices, pad),
                torch.nn.functional.pad(trace.lengths, pad),
                torch.nn.functional.pad(trace.steps, pad),
            )
        assert left.pool is right.pool, "A batch must share the same request-slot pool."
        return cls(
            left.pool,
            left.slots + right.slots,
            torch.cat((left.indices, right.indices)),
            torch.cat((left.lengths, right.lengths)),
            torch.cat((left.steps, right.steps)),
        )

    def advance(self):
        return replace(self, steps=self.steps + 1)

    def apply(self, sampled, *, inplace=False):
        return trace_replay_op(
            self.pool.prepare(self),
            self.indices,
            self.lengths,
            self.steps,
            sampled,
            inplace=inplace,
        )
