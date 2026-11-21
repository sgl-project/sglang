"""Sampling-penalty history: output-token counts and lengths, one table row per
penalized request, addressed by the request's penalty index.

- Host calls (`alloc`, `reset_history`, `free`, `StepPenalties.build`) only queue
  writes; the tables are written, grown, and read on the sampling stream.
- A request takes an index at its first extend admission; every extend admission
  resets its row and replays `output_ids`. A retraction keeps the index;
  `RequestRetirer` frees it at exit.
- A step reads and commits the indices it was built with. No request is admitted
  between a step's build and its sampling, so none of them is reused before then.
- Freeing the last live index drops the tables at once. That is safe because they
  were allocated on the sampling stream, so their memory goes only to later
  allocations there. A step whose indices were all freed sees no live index:
  it reads zeros and commits nothing.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.utils.device import async_h2d, get_compiler_backend, is_npu

_is_npu = is_npu()


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def apply_scaling_penalties(logits, scaling_penalties):
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )


def _penalty_kinds(sampling_params) -> Tuple[bool, bool, bool, bool]:
    return (
        sampling_params.frequency_penalty != 0.0,
        sampling_params.presence_penalty != 0.0,
        sampling_params.repetition_penalty != 1.0,
        sampling_params.min_new_tokens > 0,
    )


def uses_penalty(sampling_params) -> bool:
    return any(_penalty_kinds(sampling_params))


class PenaltyStatusPool:
    def __init__(self, vocab_size: int, device):
        self.vocab_size = vocab_size
        self.device = device
        self.num_indices = 0
        self.free_indices: List[int] = []
        self.pending_init: Dict[int, List[int]] = {}
        self.counts = torch.zeros((0, vocab_size), dtype=torch.int32, device=device)
        self.lens = torch.zeros((0,), dtype=torch.int32, device=device)

    def alloc(self, reqs: Sequence) -> None:
        for req in reqs:
            if req.penalty_index is not None or not uses_penalty(req.sampling_params):
                continue
            if self.free_indices:
                req.penalty_index = self.free_indices.pop()
            else:
                req.penalty_index = self.num_indices
                self.num_indices += 1

    def reset_history(self, reqs: Sequence) -> None:
        for req in reqs:
            if req.penalty_index is not None:
                self.pending_init[req.penalty_index] = list(req.output_ids)

    def free(self, req) -> None:
        if self.num_indices == 0:
            return
        index = req.penalty_index
        if index is None:
            return
        req.penalty_index = None
        self.pending_init.pop(index, None)
        self.free_indices.append(index)
        if len(self.free_indices) == self.num_indices:
            self.num_indices = 0
            self.free_indices.clear()
            self.counts = torch.empty(
                (0, self.vocab_size), dtype=torch.int32, device=self.device
            )
            self.lens = torch.empty((0,), dtype=torch.int32, device=self.device)

    def _flush(self) -> None:
        capacity = self.counts.shape[0]
        if capacity < self.num_indices:
            new_capacity = max(self.num_indices, 2 * capacity)
            counts = torch.zeros(
                (new_capacity, self.vocab_size), dtype=torch.int32, device=self.device
            )
            counts[:capacity] = self.counts
            lens = torch.zeros((new_capacity,), dtype=torch.int32, device=self.device)
            lens[:capacity] = self.lens
            self.counts, self.lens = counts, lens

        if not self.pending_init:
            return
        indices = list(self.pending_init)
        histories = list(self.pending_init.values())
        self.pending_init.clear()
        indices_device = async_h2d(indices, dtype=torch.int64, device=self.device)
        self.counts[indices_device] = 0
        self.lens[indices_device] = async_h2d(
            [len(h) for h in histories], dtype=torch.int32, device=self.device
        )
        replay_indices = [index for index, h in zip(indices, histories) for _ in h]
        if replay_indices:
            replay_tokens = [token for h in histories for token in h]
            self.counts.index_put_(
                (
                    async_h2d(replay_indices, dtype=torch.int64, device=self.device),
                    async_h2d(replay_tokens, dtype=torch.int64, device=self.device),
                ),
                torch.ones(len(replay_indices), dtype=torch.int32, device=self.device),
                accumulate=True,
            )

    def gather_counts(self, indices: torch.Tensor) -> torch.Tensor:
        self._flush()
        if self.num_indices == 0:
            return torch.zeros(
                (indices.shape[0], self.vocab_size),
                dtype=torch.int32,
                device=self.device,
            )
        return self.counts[indices]

    def gather_lens(self, indices: torch.Tensor) -> torch.Tensor:
        self._flush()
        if self.num_indices == 0:
            return torch.zeros(
                (indices.shape[0],), dtype=torch.int32, device=self.device
            )
        return self.lens[indices]

    def accumulate(
        self,
        indices: torch.Tensor,
        token_ids: torch.Tensor,
        accept_mask: Optional[torch.Tensor],
    ) -> None:
        self._flush()
        if self.num_indices == 0:
            return
        token_ids = token_ids.view(indices.shape[0], -1).long()
        # A NaN-logit sample or a masked draft slot can hold an id outside the vocab.
        counted = (token_ids >= 0) & (token_ids < self.vocab_size)
        if accept_mask is not None:
            counted &= accept_mask.view(token_ids.shape)
        weights = counted.to(torch.int32)
        token_ids = token_ids.clamp(0, self.vocab_size - 1)
        if token_ids.shape[1] == 1:
            # Indices are distinct, so no (index, token) pair repeats: a plain add is exact.
            self.counts[indices, token_ids.view(-1)] += weights.view(-1)
        else:
            self.counts.index_put_(
                (
                    indices.unsqueeze(1).expand_as(token_ids).reshape(-1),
                    token_ids.reshape(-1),
                ),
                weights.reshape(-1),
                accumulate=True,
            )
        self.lens.index_add_(0, indices, weights.sum(dim=1, dtype=torch.int32))


@dataclasses.dataclass
class StepPenalties:
    pool: PenaltyStatusPool
    batch_size: int
    # [n] batch positions and penalty indices of the n penalized requests.
    batch_indices: torch.Tensor
    penalty_indices: torch.Tensor
    # [n, 1] per penalized request, None when none of them uses that penalty.
    frequency: Optional[torch.Tensor]
    presence: Optional[torch.Tensor]
    repetition: Optional[torch.Tensor]
    min_new_tokens: Optional[torch.Tensor]
    # [n, k] stop token ids, padded with vocab_size; set with min_new_tokens.
    stop_token_ids: Optional[torch.Tensor]
    _tables: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = (
        dataclasses.field(default=None, init=False, repr=False, compare=False)
    )

    @classmethod
    def build(cls, reqs: Sequence, pool: PenaltyStatusPool) -> Optional[StepPenalties]:
        batch_indices = [i for i, r in enumerate(reqs) if r.penalty_index is not None]
        if not batch_indices:
            return None
        penalty_indices = [reqs[i].penalty_index for i in batch_indices]

        device = pool.device
        params = [reqs[i].sampling_params for i in batch_indices]
        uses_frequency, uses_presence, uses_repetition, uses_min_new_tokens = (
            any(kind) for kind in zip(*(_penalty_kinds(p) for p in params))
        )

        def per_request(values, dtype, needed):
            if not needed:
                return None
            return async_h2d(values, dtype=dtype, device=device).unsqueeze_(1)

        stop_token_ids = None
        if uses_min_new_tokens:
            stop_sets = [
                _stop_token_ids(reqs[i], pool.vocab_size) for i in batch_indices
            ]
            width = max(1, max(len(s) for s in stop_sets))
            padded = [s + [pool.vocab_size] * (width - len(s)) for s in stop_sets]
            stop_token_ids = async_h2d(padded, dtype=torch.int64, device=device)

        return cls(
            pool=pool,
            batch_size=len(reqs),
            batch_indices=async_h2d(batch_indices, dtype=torch.int64, device=device),
            penalty_indices=async_h2d(
                penalty_indices, dtype=torch.int64, device=device
            ),
            frequency=per_request(
                [p.frequency_penalty for p in params], torch.float32, uses_frequency
            ),
            presence=per_request(
                [p.presence_penalty for p in params], torch.float32, uses_presence
            ),
            repetition=per_request(
                [p.repetition_penalty for p in params], torch.float32, uses_repetition
            ),
            min_new_tokens=per_request(
                [p.min_new_tokens for p in params], torch.int32, uses_min_new_tokens
            ),
            stop_token_ids=stop_token_ids,
        )

    def logit_penalties(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """(additive, scaling) [batch_size, vocab] penalties, each None if unused;
        built on the first call, which must run on the sampling stream."""
        if self._tables is None:
            self._tables = self._build_tables()
        return self._tables

    def _build_tables(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        counts = self.pool.gather_counts(self.penalty_indices)
        vocab_size = self.pool.vocab_size
        device = counts.device
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
            if self.frequency is not None:
                additive.addcmul_(counts, self.frequency, value=-1)
            if self.presence is not None:
                additive.addcmul_(present, self.presence, value=-1)
            if self.min_new_tokens is not None:
                lens = self.pool.gather_lens(self.penalty_indices)
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

    def _to_batch(self, values: torch.Tensor, fill: float) -> torch.Tensor:
        if self.batch_indices.numel() == self.batch_size:
            return values
        out = torch.full(
            (self.batch_size, values.shape[1]),
            fill,
            dtype=values.dtype,
            device=values.device,
        )
        out[self.batch_indices] = values
        return out

    def commit(
        self, token_ids: torch.Tensor, accept_mask: Optional[torch.Tensor] = None
    ) -> None:
        """token_ids: [batch_size] sampled ids, or [batch_size, k] verify tokens
        with `accept_mask` marking the ones that count."""
        token_ids = token_ids.view(self.batch_size, -1)
        if accept_mask is not None:
            accept_mask = accept_mask.view(self.batch_size, -1)
        if self.batch_indices.numel() != self.batch_size:
            token_ids = token_ids[self.batch_indices]
            if accept_mask is not None:
                accept_mask = accept_mask[self.batch_indices]
        self.pool.accumulate(self.penalty_indices, token_ids, accept_mask)


def _stop_token_ids(req, vocab_size: int) -> List[int]:
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
    # Out-of-vocab ids would index outside the penalty tensor.
    return [
        token_id
        for token_id in stop_ids
        if token_id is not None and 0 <= token_id < vocab_size
    ]
