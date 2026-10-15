"""Penalty history in per-request rows, found through a slot -> row table.

`PenaltyStore` is one per scheduler: `alloc` gives a penalized request a row
on `req.kv.penalty_row` and (re)sets it to the history of its `output_ids`
each time the request enters an extend step; the request's release path
(`free_req_slot`) frees the row with the request's slot.

Invariants:
- Only a request whose params use a penalty holds a row; a step without one
  has no `StepPenalties`.
- A row holds history only (per-token output counts and the output length);
  penalty values come from the request's params at build.
- Host calls (`alloc`, `free`, `build`) update only host state and queue
  device writes; the table and the rows are written, grown, reset, and read
  only on the sampling stream, so a step sees exactly the tokens of the steps
  sampled before it.
- A step finds its rows in the table at sampling, from its slots. `free`
  leaves the table entry, so a step still holding a freed slot (a finished
  request's last decode under overlap) reads the row it was built with; a
  slot is not reallocated between the build of a step holding it and that
  step's sampling. A row freed in between may still take the step's commit,
  and its next `alloc` resets it.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.utils import get_compiler_backend, is_npu
from sglang.srt.utils.common import is_pin_memory_available

_is_npu = is_npu()


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def apply_scaling_penalties(logits, scaling_penalties):
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )


def uses_penalty(sampling_params) -> bool:
    return (
        sampling_params.frequency_penalty != 0.0
        or sampling_params.presence_penalty != 0.0
        or sampling_params.repetition_penalty != 1.0
        or sampling_params.min_new_tokens > 0
    )


def _to_device(data, dtype: torch.dtype, device, pin: bool) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=pin).to(device, non_blocking=True)


@dataclasses.dataclass(frozen=True, slots=True)
class PenaltyRow:
    """A request's row in its store; `req.kv` holds it from `alloc` until the
    request's slot is freed."""

    store: PenaltyStore
    index: int


class PenaltyStore:
    def __init__(self, vocab_size: int, num_slots: int, device):
        self.vocab_size = vocab_size
        self.device = device
        self.pin = is_pin_memory_available(device)
        self.num_rows = 0
        self.free_rows: List[int] = []
        # Table entries to write before their next device use.
        self.pending_slots: Dict[int, int] = {}
        # Rows to reset before their next device use, with the tokens to replay.
        self.pending_init: Dict[int, List[int]] = {}
        # The row of each request slot, like req_to_token: -1 until the slot's
        # first penalized request; a freed slot keeps its entry until the next
        # alloc into it.
        self.slot_to_row = torch.full(
            (num_slots,), -1, dtype=torch.int32, device=device
        )
        self.counts = torch.zeros((0, vocab_size), dtype=torch.int32, device=device)
        self.lens = torch.zeros((0,), dtype=torch.int32, device=device)

    def alloc(self, req) -> None:
        if not uses_penalty(req.sampling_params):
            return
        kv = req.kv
        if kv.penalty_row is None:
            if self.free_rows:
                index = self.free_rows.pop()
            else:
                index = self.num_rows
                self.num_rows += 1
            kv.penalty_row = PenaltyRow(self, index)
        row = kv.penalty_row.index
        self.pending_slots[kv.req_pool_idx] = row
        self.pending_init[row] = list(req.output_ids)

    def free(self, kv) -> None:
        row = kv.penalty_row.index
        kv.penalty_row = None
        self.pending_init.pop(row, None)
        self.free_rows.append(row)

    def _flush(self) -> None:
        capacity = self.counts.shape[0]
        if capacity < self.num_rows:
            new_capacity = max(self.num_rows, 2 * capacity)
            counts = torch.zeros(
                (new_capacity, self.vocab_size), dtype=torch.int32, device=self.device
            )
            counts[:capacity] = self.counts
            lens = torch.zeros((new_capacity,), dtype=torch.int32, device=self.device)
            lens[:capacity] = self.lens
            self.counts, self.lens = counts, lens

        if self.pending_slots:
            slots = list(self.pending_slots)
            rows = list(self.pending_slots.values())
            self.pending_slots.clear()
            self.slot_to_row[_to_device(slots, torch.int64, self.device, self.pin)] = (
                _to_device(rows, torch.int32, self.device, self.pin)
            )

        if not self.pending_init:
            return
        rows = list(self.pending_init)
        histories = list(self.pending_init.values())
        self.pending_init.clear()
        rows_device = _to_device(rows, torch.int64, self.device, self.pin)
        self.counts[rows_device] = 0
        self.lens[rows_device] = _to_device(
            [len(h) for h in histories], torch.int32, self.device, self.pin
        )
        replay_rows = [row for row, h in zip(rows, histories) for _ in h]
        if replay_rows:
            replay_tokens = [token for h in histories for token in h]
            self.counts.index_put_(
                (
                    _to_device(replay_rows, torch.int64, self.device, self.pin),
                    _to_device(replay_tokens, torch.int64, self.device, self.pin),
                ),
                torch.ones(len(replay_rows), dtype=torch.int32, device=self.device),
                accumulate=True,
            )

    def rows(self, slots: torch.Tensor) -> torch.Tensor:
        """The rows of the requests in the slots."""
        self._flush()
        return self.slot_to_row[slots]

    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        """The rows' token counts."""
        self._flush()
        return self.counts[rows]

    def gather_lens(self, rows: torch.Tensor) -> torch.Tensor:
        """The rows' output lengths."""
        self._flush()
        return self.lens[rows]

    def accumulate(
        self,
        rows: torch.Tensor,
        token_ids: torch.Tensor,
        accept_mask: Optional[torch.Tensor],
    ) -> None:
        """Count token_ids[i, j] into rows[i] where accept_mask[i, j] (all if None)."""
        self._flush()
        token_ids = token_ids.view(rows.shape[0], -1).long()
        if accept_mask is None and token_ids.shape[1] == 1:
            # One token per row over distinct rows: no (row, token) pair
            # repeats, so a plain add counts them without a sorted accumulate.
            self.counts[rows, token_ids.view(-1)] += 1
            self.lens.index_add_(0, rows, torch.ones_like(rows, dtype=torch.int32))
            return
        if accept_mask is None:
            weights = torch.ones_like(token_ids, dtype=torch.int32)
        else:
            weights = accept_mask.view(token_ids.shape).to(torch.int32)
        self.counts.index_put_(
            (rows.unsqueeze(1).expand_as(token_ids).reshape(-1), token_ids.reshape(-1)),
            weights.reshape(-1),
            accumulate=True,
        )
        self.lens.index_add_(0, rows, weights.sum(dim=1, dtype=torch.int32))


@dataclasses.dataclass(frozen=True)
class StepPenalties:
    """The penalized requests of one step: their slots and penalty params."""

    store: PenaltyStore
    batch_size: int
    # [n] step positions of the penalized requests, and their slots; their rows
    # are read from the store's table at sampling.
    positions: torch.Tensor
    slots: torch.Tensor
    # [n, 1] per penalized request, None when none of them uses that penalty.
    frequency: Optional[torch.Tensor]
    presence: Optional[torch.Tensor]
    repetition: Optional[torch.Tensor]
    min_new_tokens: Optional[torch.Tensor]
    # [n, k] stop token ids, padded with vocab_size; set with min_new_tokens.
    stop_token_ids: Optional[torch.Tensor]

    @classmethod
    def build(
        cls,
        reqs: Sequence,
        req_pool_indices: Optional[torch.Tensor],
        store: Optional[PenaltyStore],
        device,
    ) -> Optional[StepPenalties]:
        """req_pool_indices: the step's slots on the device, in request order."""
        positions = [i for i, r in enumerate(reqs) if uses_penalty(r.sampling_params)]
        if not positions:
            return None
        if store is None or req_pool_indices is None:
            raise RuntimeError(
                "requests use sampling penalties but no PenaltyStore or slots"
            )
        for i in positions:
            if reqs[i].kv.penalty_row is None:
                raise RuntimeError(
                    f"request {reqs[i].rid} uses sampling penalties but holds no "
                    "penalty row; PenaltyStore.alloc was not called"
                )

        pin = is_pin_memory_available(device)
        positions_device = _to_device(positions, torch.int64, device, pin)
        if len(positions) == len(reqs):
            slots = req_pool_indices[: len(reqs)]
        else:
            slots = req_pool_indices[positions_device]
        params = [reqs[i].sampling_params for i in positions]

        def per_row(values, dtype, needed):
            if not needed:
                return None
            return _to_device(values, dtype, device, pin).unsqueeze_(1)

        frequency = per_row(
            [p.frequency_penalty for p in params],
            torch.float32,
            any(p.frequency_penalty != 0.0 for p in params),
        )
        presence = per_row(
            [p.presence_penalty for p in params],
            torch.float32,
            any(p.presence_penalty != 0.0 for p in params),
        )
        repetition = per_row(
            [p.repetition_penalty for p in params],
            torch.float32,
            any(p.repetition_penalty != 1.0 for p in params),
        )
        needs_min_new_tokens = any(p.min_new_tokens > 0 for p in params)
        min_new_tokens = per_row(
            [p.min_new_tokens for p in params], torch.int32, needs_min_new_tokens
        )
        stop_token_ids = None
        if needs_min_new_tokens:
            stop_sets = [_stop_token_ids(reqs[i]) for i in positions]
            width = max(1, max(len(s) for s in stop_sets))
            padded = [s + [store.vocab_size] * (width - len(s)) for s in stop_sets]
            stop_token_ids = _to_device(padded, torch.int64, device, pin)

        return cls(
            store=store,
            batch_size=len(reqs),
            positions=positions_device,
            slots=slots,
            frequency=frequency,
            presence=presence,
            repetition=repetition,
            min_new_tokens=min_new_tokens,
            stop_token_ids=stop_token_ids,
        )

    def logit_penalties(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (additive, scaling) [batch_size, vocab] penalties, each None if unused."""
        rows = self.store.rows(self.slots)
        counts = self.store.gather(rows)
        vocab_size = self.store.vocab_size
        device = counts.device
        # The tokens each row has output, for presence and repetition.
        present = (
            counts > 0
            if self.presence is not None or self.repetition is not None
            else None
        )

        additive = None
        if (
            self.frequency is not None
            or self.presence is not None
            or self.min_new_tokens is not None
        ):
            additive = torch.zeros(counts.shape, dtype=torch.float32, device=device)
            # addcmul_ with value=-1 subtracts a product in one kernel; on these
            # operands it rounds as the product and the subtraction would.
            if self.frequency is not None:
                additive.addcmul_(counts, self.frequency, value=-1)
            if self.presence is not None:
                additive.addcmul_(present, self.presence, value=-1)
            if self.min_new_tokens is not None:
                lens = self.store.gather_lens(rows)
                stop_penalties = torch.zeros(
                    (counts.shape[0], vocab_size + 1),
                    dtype=torch.float32,
                    device=device,
                ).scatter_(1, self.stop_token_ids, float("-inf"))[:, :vocab_size]
                # torch.where, not a boolean index: no host sync, no -inf * 0.
                additive.add_(
                    torch.where(
                        lens.unsqueeze(1) < self.min_new_tokens, stop_penalties, 0.0
                    )
                )
            additive = self._to_batch(additive, 0.0)

        scaling = None
        if self.repetition is not None:
            scaling = torch.where(present, self.repetition, 1.0)
            scaling = self._to_batch(scaling, 1.0)
        return additive, scaling

    def _to_batch(self, rows: torch.Tensor, fill: float) -> torch.Tensor:
        if self.positions.numel() == self.batch_size:
            return rows
        out = torch.full(
            (self.batch_size, rows.shape[1]), fill, dtype=rows.dtype, device=rows.device
        )
        out[self.positions] = rows
        return out

    def commit(
        self, token_ids: torch.Tensor, accept_mask: Optional[torch.Tensor] = None
    ) -> None:
        """Count this step's output tokens into the rows of its requests.

        token_ids: [batch_size] sampled ids, or [batch_size, k] with `accept_mask`
        marking the ids that count (speculative verify).
        """
        token_ids = token_ids.view(self.batch_size, -1)
        if accept_mask is not None:
            accept_mask = accept_mask.view(self.batch_size, -1)
        if self.positions.numel() != self.batch_size:
            # Only some requests are penalized: take their rows of the step.
            token_ids = token_ids[self.positions]
            if accept_mask is not None:
                accept_mask = accept_mask[self.positions]
        self.store.accumulate(self.store.rows(self.slots), token_ids, accept_mask)


def _stop_token_ids(req) -> List[int]:
    stop_ids = (
        (req.sampling_params.stop_token_ids or set())
        | (req.eos_token_ids or set())
        | (req.tokenizer.additional_stop_token_ids or set())
        | (
            {req.tokenizer.eos_token_id}
            if req.tokenizer.eos_token_id is not None
            else set()
        )
    )
    return [token_id for token_id in stop_ids if token_id is not None]
