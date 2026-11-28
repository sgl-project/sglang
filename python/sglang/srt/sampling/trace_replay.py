"""Compact device storage and batch metadata for trace replay."""

import hashlib
from array import array
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Optional

import torch

from sglang.kernels.ops.sampling import trace_replay as trace_replay_op
from sglang.srt.utils.device import (
    async_h2d,
    get_device_module,
    is_pin_memory_available,
)


def _trace_fingerprint(token_ids: Sequence[int], length: int) -> bytes:
    """Compute a 128-bit digest of the active trace prefix without retaining it."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(array("q", token_ids[:length]).tobytes())
    return digest.digest()


@dataclass(frozen=True)
class TraceReplaySlot:
    index: int
    generation: int
    fingerprint: bytes
    # The tokenizer publishes a request-local list. Scheduling may cap the
    # effective length, but never mutates the published token sequence.
    token_ids: list[int]
    length: int

    @property
    def key(self) -> tuple[int, int, bytes, int]:
        return (self.index, self.generation, self.fingerprint, self.length)


class TraceReplayPool:
    """Compact device token table storing rows only for active traced requests.

    Row 0 is reserved for non-trace entries. Active requests are mapped to
    compact 1-based rows keyed by ``(req_pool_idx, generation, fingerprint,
    length)``, so neither device memory nor host residency metadata scales with
    the full request-pool capacity or retains finished token lists. Rows are
    uploaded on the consuming stream, and a delayed sampler can restore its own
    trace from its snapshot if a newer request reclaimed the row.
    """

    def __init__(self, req_pool):
        self.device = torch.device(req_pool.device)
        self._req_generation = req_pool.req_generation
        self.max_rows = req_pool.req_generation.numel()
        self.max_width = req_pool.max_context_len
        self.tokens: Optional[torch.Tensor] = None
        self._key_to_row: dict[tuple[int, int, bytes, int], int] = {}
        self._row_to_key: dict[int, tuple[int, int, bytes, int]] = {}
        self._row_last_used: dict[int, int] = {}
        self._use_tick = 0
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

    def _ensure_capacity(self, rows: int, width: int):
        if rows > self.max_rows:
            raise ValueError(
                "Active trace count exceeds the request pool's slot capacity."
            )
        if width > self.max_width:
            raise ValueError(
                "Trace length exceeds the request pool's context capacity."
            )
        old = self.tokens
        old_rows, old_width = old.shape if old is not None else (0, 0)
        if rows <= old_rows and width <= old_width:
            return
        new_rows = (
            old_rows
            if rows <= old_rows
            else (
                rows if old_rows == 0 else min(self.max_rows, max(rows, 2 * old_rows))
            )
        )
        new_width = (
            old_width
            if width <= old_width
            else (
                width
                if old_width == 0
                else min(self.max_width, max(width, 2 * old_width))
            )
        )
        self.tokens = torch.empty(
            (new_rows, new_width), dtype=torch.int32, device=self.device
        )
        if old is not None:
            self.tokens[:old_rows, :old_width].copy_(old)
        else:
            self.tokens[0, 0] = 0

    def _evict_row(self, row: int) -> None:
        old_key = self._row_to_key.pop(row)
        self._key_to_row.pop(old_key, None)
        self._row_last_used.pop(row, None)

    def prepare(self, batch):
        self._use_current_stream()
        self._use_tick += 1
        if (
            batch._prepared_revision == self.revision
            and batch._prepared_rows is not None
        ):
            for slot in batch.slots:
                if slot is not None:
                    row = self._key_to_row.get(slot.key)
                    if row is not None:
                        self._row_last_used[row] = self._use_tick
            return self.tokens, batch._prepared_rows

        active_keys = {slot.key for slot in batch.slots if slot is not None}
        active_indices = {slot.index for slot in batch.slots if slot is not None}

        # Prefer reclaiming rows whose request slot moved to a new generation
        # or was reused by a new streaming-session turn, then unused allocated
        # rows, then growing the table before evicting any live LRU row from
        # another concurrent batch.
        dead_rows: list[int] = []
        live_other_rows: list[int] = []
        for row, key in self._row_to_key.items():
            if key in active_keys:
                continue
            req_idx, gen, _, _ = key
            if int(self._req_generation[req_idx]) != gen or req_idx in active_indices:
                dead_rows.append(row)
            else:
                live_other_rows.append(row)
        live_other_rows.sort(
            key=lambda row: self._row_last_used.get(row, 0), reverse=True
        )

        allocated_rows = self.tokens.shape[0] if self.tokens is not None else 1
        free_rows = [
            row for row in range(1, allocated_rows) if row not in self._row_to_key
        ]
        next_row = allocated_rows

        pending: list[tuple[int, TraceReplaySlot]] = []
        assigned: dict[tuple[int, int, bytes, int], int] = {}
        for slot in batch.slots:
            if slot is None:
                continue
            key = slot.key
            if key in assigned:
                continue
            row = self._key_to_row.get(key)
            if row is not None:
                assigned[key] = row
                self._row_last_used[row] = self._use_tick
                continue
            if dead_rows:
                row = dead_rows.pop()
                self._evict_row(row)
            elif free_rows:
                row = free_rows.pop()
            elif next_row < self.max_rows:
                row = next_row
                next_row += 1
            else:
                row = live_other_rows.pop()
                self._evict_row(row)
            self._key_to_row[key] = row
            self._row_to_key[row] = key
            self._row_last_used[row] = self._use_tick
            assigned[key] = row
            pending.append((row, slot))

        if pending:
            needed_rows = max(next_row, max(row for row, _ in pending) + 1)
            needed_width = max(slot.length for _, slot in pending)
            self._ensure_capacity(needed_rows, needed_width)
            pin = is_pin_memory_available(self.device)
            for row, slot in pending:
                host = torch.tensor(
                    slot.token_ids[: slot.length], dtype=torch.int32, pin_memory=pin
                )
                self.tokens[row, : slot.length].copy_(host, non_blocking=pin)
            self.revision += 1

        row_indices = [
            assigned[slot.key] if slot is not None else 0 for slot in batch.slots
        ]
        batch._prepared_rows = async_h2d(
            row_indices, dtype=torch.int32, device=self.device
        )
        batch._prepared_revision = self.revision
        return self.tokens, batch._prepared_rows


@dataclass
class TraceReplayBatch:
    pool: TraceReplayPool
    slots: tuple[Optional[TraceReplaySlot], ...]
    lengths: torch.Tensor
    steps: torch.Tensor
    # Host/device residency cache updated by prepare() on the consumer stream.
    # In-flight forward snapshots retain their slot bindings and step tensors.
    _prepared_rows: Optional[torch.Tensor] = None
    _prepared_revision: int = -1

    @classmethod
    def from_requests(cls, reqs, req_pool):
        traces = [
            getattr(req.sampling_params, "trace_decode_token_ids", None) for req in reqs
        ]
        if not any(
            isinstance(trace, Sequence)
            and not isinstance(trace, (str, bytes))
            and len(trace) > 0
            for trace in traces
        ):
            return None

        slots = []
        for req, trace in zip(reqs, traces, strict=True):
            if (
                not isinstance(trace, Sequence)
                or isinstance(trace, (str, bytes))
                or len(trace) == 0
            ):
                slots.append(None)
                continue
            token_list = trace if isinstance(trace, list) else list(trace)
            cap = req.sampling_params.max_new_tokens
            length = len(token_list) if cap is None else min(len(token_list), cap)
            if length <= 0:
                slots.append(None)
                continue
            index = req.kv.req_pool_idx
            assert index is not None and index > 0
            slots.append(
                TraceReplaySlot(
                    index,
                    int(req_pool.req_generation[index]),
                    _trace_fingerprint(token_list, length),
                    token_list,
                    length,
                )
            )

        if not any(slot is not None for slot in slots):
            return None
        pool = getattr(req_pool, "trace_replay_pool", None)
        if pool is None:
            pool = req_pool.trace_replay_pool = TraceReplayPool(req_pool)
        meta = async_h2d(
            [
                [slot.length if slot is not None else 0 for slot in slots],
                [
                    len(req.output_ids) if slot is not None else 0
                    for req, slot in zip(reqs, slots, strict=True)
                ],
            ],
            dtype=torch.int32,
            device=pool.device,
        )
        return cls(pool, tuple(slots), meta[0], meta[1])

    def filter(self, keep_indices, keep_indices_device):
        slots = tuple(self.slots[i] for i in keep_indices)
        if not any(slot is not None for slot in slots):
            return None
        # Do not slice _prepared_rows on the schedule stream: it is written by
        # prepare() on the forward stream, so leaving it None lets prepare()
        # rebuild the compact row indices safely on the consumer stream.
        return TraceReplayBatch(
            self.pool,
            slots,
            self.lengths[keep_indices_device],
            self.steps[keep_indices_device],
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
                torch.nn.functional.pad(trace.lengths, pad),
                torch.nn.functional.pad(trace.steps, pad),
            )
        assert left.pool is right.pool, "A batch must share the same request-slot pool."
        return cls(
            left.pool,
            left.slots + right.slots,
            torch.cat((left.lengths, right.lengths)),
            torch.cat((left.steps, right.steps)),
        )

    def advance(self):
        return replace(self, steps=self.steps + 1)

    def apply(self, sampled, *, inplace=False):
        table, rows = self.pool.prepare(self)
        return trace_replay_op(
            table,
            rows,
            self.lengths,
            self.steps,
            sampled,
            inplace=inplace,
        )
