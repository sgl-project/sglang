"""Sampling-penalty state: per-token output counts and lengths plus the
request's penalty parameters, one table row per penalized request, addressed by
its penalty index.

- Host calls (`alloc`, `reset_history`, `free`, `StepPenalties.build`) only queue
  writes; the tables are written, grown, and read on the sampling stream.
- A request takes an index at its first extend admission; every extend admission
  resets its history and replays `output_ids`. A retraction keeps the index;
  `RequestRetirer` frees it at exit.
- A step reads and commits the indices it was built with. No request is admitted
  between a step's build and its sampling, so none of them is reused before then.
- Freeing the last live index drops the tables at once; their memory is from the
  sampling stream, so it goes only to later allocations there. A step whose
  indices were all freed applies no penalty and commits nothing.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from sglang.srt.utils.device import (
    async_h2d,
    get_compiler_backend,
    get_device_sm,
    is_cuda,
    is_npu,
)

_is_cuda = is_cuda()
# The verify kernel stages rows with cp.async, which needs sm_80.
_has_verify_kernel = _is_cuda and get_device_sm() >= 80
_is_npu = is_npu()

_FREQUENCY = 1
_PRESENCE = 2
_REPETITION = 4
_MIN_NEW_TOKENS = 8


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def apply_scaling_penalties(logits, scaling_penalties):
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )


def _penalty_kinds(sampling_params) -> int:
    kinds = 0
    if sampling_params.frequency_penalty != 0.0:
        kinds |= _FREQUENCY
    if sampling_params.presence_penalty != 0.0:
        kinds |= _PRESENCE
    if sampling_params.repetition_penalty != 1.0:
        kinds |= _REPETITION
    if sampling_params.min_new_tokens > 0:
        kinds |= _MIN_NEW_TOKENS
    return kinds


def uses_penalty(sampling_params) -> bool:
    return _penalty_kinds(sampling_params) != 0


class PenaltyTables(NamedTuple):
    """One row per penalty index."""

    counts: torch.Tensor  # [rows, vocab] int32 output-token counts
    lens: torch.Tensor  # [rows] int32 output lengths
    frequency: torch.Tensor  # [rows] fp32
    presence: torch.Tensor  # [rows] fp32
    repetition: torch.Tensor  # [rows] fp32
    min_new_tokens: torch.Tensor  # [rows] int32
    # [rows, k] int64, padded with vocab_size; empty unless min_new_tokens > 0.
    stop_token_ids: torch.Tensor


class _RequestParams(NamedTuple):
    frequency: float
    presence: float
    repetition: float
    min_new_tokens: int
    stop_token_ids: List[int]


def _grown(table: torch.Tensor, shape: Tuple[int, ...], fill) -> torch.Tensor:
    out = table.new_full(shape, fill)
    out[tuple(slice(n) for n in table.shape)] = table
    return out


class PenaltyStatusPool:
    def __init__(self, vocab_size: int, device):
        self.vocab_size = vocab_size
        self.device = device
        self.num_indices = 0
        self.free_indices: List[int] = []
        self.kinds: List[int] = []
        self.pending_params: Dict[int, _RequestParams] = {}
        self.pending_init: Dict[int, List[int]] = {}
        self.tables = self._empty_tables()

    def _empty_tables(self) -> PenaltyTables:
        def empty(*shape, dtype):
            return torch.empty((0, *shape), dtype=dtype, device=self.device)

        return PenaltyTables(
            counts=empty(self.vocab_size, dtype=torch.int32),
            lens=empty(dtype=torch.int32),
            frequency=empty(dtype=torch.float32),
            presence=empty(dtype=torch.float32),
            repetition=empty(dtype=torch.float32),
            min_new_tokens=empty(dtype=torch.int32),
            stop_token_ids=empty(1, dtype=torch.int64),
        )

    def alloc(self, reqs: Sequence) -> None:
        for req in reqs:
            if req.penalty_index is not None:
                continue
            kinds = _penalty_kinds(req.sampling_params)
            if not kinds:
                continue
            if self.free_indices:
                index = self.free_indices.pop()
                self.kinds[index] = kinds
            else:
                index = self.num_indices
                self.num_indices += 1
                self.kinds.append(kinds)
            req.penalty_index = index
            params = req.sampling_params
            self.pending_params[index] = _RequestParams(
                frequency=params.frequency_penalty,
                presence=params.presence_penalty,
                repetition=params.repetition_penalty,
                min_new_tokens=params.min_new_tokens,
                stop_token_ids=(
                    _stop_token_ids(req, self.vocab_size)
                    if kinds & _MIN_NEW_TOKENS
                    else []
                ),
            )

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
        self.pending_params.pop(index, None)
        self.pending_init.pop(index, None)
        self.free_indices.append(index)
        if len(self.free_indices) == self.num_indices:
            self.num_indices = 0
            self.free_indices.clear()
            self.kinds.clear()
            self.tables = self._empty_tables()

    def _flush(self) -> None:
        capacity, stop_width = self.tables.stop_token_ids.shape
        if capacity < self.num_indices:
            new_capacity = max(self.num_indices, 2 * capacity)

            def grown(table, fill):
                return _grown(table, (new_capacity, *table.shape[1:]), fill)

            tables = self.tables
            self.tables = PenaltyTables(
                counts=grown(tables.counts, 0),
                lens=grown(tables.lens, 0),
                frequency=grown(tables.frequency, 0.0),
                presence=grown(tables.presence, 0.0),
                repetition=grown(tables.repetition, 1.0),
                min_new_tokens=grown(tables.min_new_tokens, 0),
                stop_token_ids=grown(tables.stop_token_ids, self.vocab_size),
            )
        needed_stop_width = max(
            (len(p.stop_token_ids) for p in self.pending_params.values()), default=0
        )
        if stop_width < needed_stop_width:
            stop_token_ids = self.tables.stop_token_ids
            self.tables = self.tables._replace(
                stop_token_ids=_grown(
                    stop_token_ids,
                    (stop_token_ids.shape[0], max(needed_stop_width, 2 * stop_width)),
                    self.vocab_size,
                )
            )

        if self.pending_params:
            self._write_params()
        if self.pending_init:
            self._replay_histories()

    def _write_params(self) -> None:
        tables = self.tables
        indices = list(self.pending_params)
        params = list(self.pending_params.values())
        self.pending_params.clear()
        stop_width = tables.stop_token_ids.shape[1]
        coefficients_cpu = [[p.frequency, p.presence, p.repetition] for p in params]
        stop_token_ids_cpu = [
            p.stop_token_ids + [self.vocab_size] * (stop_width - len(p.stop_token_ids))
            for p in params
        ]
        indices_device = async_h2d(indices, dtype=torch.int64, device=self.device)
        coefficients = async_h2d(
            coefficients_cpu, dtype=torch.float32, device=self.device
        )
        min_new_tokens = async_h2d(
            [p.min_new_tokens for p in params], dtype=torch.int32, device=self.device
        )
        stop_token_ids = async_h2d(
            stop_token_ids_cpu, dtype=torch.int64, device=self.device
        )
        tables.frequency[indices_device] = coefficients[:, 0]
        tables.presence[indices_device] = coefficients[:, 1]
        tables.repetition[indices_device] = coefficients[:, 2]
        tables.min_new_tokens[indices_device] = min_new_tokens
        tables.stop_token_ids[indices_device] = stop_token_ids

    def _replay_histories(self) -> None:
        tables = self.tables
        indices = list(self.pending_init)
        histories = list(self.pending_init.values())
        self.pending_init.clear()
        indices_device = async_h2d(indices, dtype=torch.int64, device=self.device)
        tables.counts[indices_device] = 0
        tables.lens[indices_device] = async_h2d(
            [len(h) for h in histories], dtype=torch.int32, device=self.device
        )
        replay_indices = [index for index, h in zip(indices, histories) for _ in h]
        if replay_indices:
            replay_tokens = [token for h in histories for token in h]
            tables.counts.index_put_(
                (
                    async_h2d(replay_indices, dtype=torch.int64, device=self.device),
                    async_h2d(replay_tokens, dtype=torch.int64, device=self.device),
                ),
                torch.ones(len(replay_indices), dtype=torch.int32, device=self.device),
                accumulate=True,
            )

    def live_tables(self) -> Optional[PenaltyTables]:
        """The tables with queued writes applied, or None when no index is live."""
        self._flush()
        if self.num_indices == 0:
            return None
        return self.tables

    def accumulate(
        self,
        batch_indices: torch.Tensor,
        indices: torch.Tensor,
        token_ids: torch.Tensor,
        accept_mask: Optional[torch.Tensor],
    ) -> None:
        """Count row `batch_indices[i]` of `token_ids` into index `indices[i]`."""
        tables = self.live_tables()
        if tables is None:
            return
        if _is_cuda and token_ids.is_cuda:
            from sglang.kernels.ops.sampling.penalty_counts import (
                accumulate_penalty_counts,
            )

            accumulate_penalty_counts(
                tables.counts,
                tables.lens,
                batch_indices,
                indices,
                token_ids,
                accept_mask,
            )
            return
        token_ids = token_ids[batch_indices].long()
        # A NaN-logit sample or a masked draft slot can hold an id outside the vocab.
        counted = (token_ids >= 0) & (token_ids < self.vocab_size)
        if accept_mask is not None:
            counted &= accept_mask[batch_indices]
        weights = counted.to(torch.int32)
        token_ids = token_ids.clamp(0, self.vocab_size - 1)
        if token_ids.shape[1] == 1:
            # Indices are distinct, so a plain add is exact.
            tables.counts[indices, token_ids.view(-1)] += weights.view(-1)
        else:
            tables.counts.index_put_(
                (
                    indices.unsqueeze(1).expand_as(token_ids).reshape(-1),
                    token_ids.reshape(-1),
                ),
                weights.reshape(-1),
                accumulate=True,
            )
        tables.lens.index_add_(0, indices, weights.sum(dim=1, dtype=torch.int32))


class _StepParams(NamedTuple):
    """Each None when no request of the step uses that penalty."""

    frequency: Optional[torch.Tensor]
    presence: Optional[torch.Tensor]
    repetition: Optional[torch.Tensor]
    min_new_tokens: Optional[torch.Tensor]
    stop_token_ids: Optional[torch.Tensor]

    def gather(self, indices: torch.Tensor) -> _StepParams:
        """[n, 1] values, or [n, k] stop ids, of the n rows at `indices`."""
        return _StepParams(
            *(
                None if t is None else t[indices].view(indices.shape[0], -1)
                for t in self
            )
        )


@dataclasses.dataclass
class StepPenalties:
    pool: PenaltyStatusPool
    batch_size: int
    batch_indices: torch.Tensor
    penalty_indices: torch.Tensor
    kinds: int
    _logit_penalties: Optional[
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
    ] = dataclasses.field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def build(cls, reqs: Sequence, pool: PenaltyStatusPool) -> Optional[StepPenalties]:
        batch_indices = [i for i, r in enumerate(reqs) if r.penalty_index is not None]
        if not batch_indices:
            return None
        penalty_indices = [reqs[i].penalty_index for i in batch_indices]
        kinds = functools.reduce(operator.or_, (pool.kinds[i] for i in penalty_indices))
        indices = async_h2d(
            [batch_indices, penalty_indices], dtype=torch.int64, device=pool.device
        )
        return cls(
            pool=pool,
            batch_size=len(reqs),
            batch_indices=indices[0],
            penalty_indices=indices[1],
            kinds=kinds,
        )

    def _params(self, tables: PenaltyTables) -> _StepParams:
        uses_min_new_tokens = bool(self.kinds & _MIN_NEW_TOKENS)
        return _StepParams(
            frequency=tables.frequency if self.kinds & _FREQUENCY else None,
            presence=tables.presence if self.kinds & _PRESENCE else None,
            repetition=tables.repetition if self.kinds & _REPETITION else None,
            min_new_tokens=tables.min_new_tokens if uses_min_new_tokens else None,
            stop_token_ids=tables.stop_token_ids if uses_min_new_tokens else None,
        )

    def logit_penalties(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """(additive, scaling) [batch_size, vocab], built on the first call, which
        must run on the sampling stream."""
        if self._logit_penalties is None:
            self._logit_penalties = self._build_logit_penalties()
        return self._logit_penalties

    def _build_logit_penalties(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        tables = self.pool.live_tables()
        if tables is None:
            # Every request of the step has exited.
            return None, None
        params = self._params(tables).gather(self.penalty_indices)
        additive, scaling = _compute_logit_penalties(
            tables.counts[self.penalty_indices],
            (
                tables.lens[self.penalty_indices]
                if params.min_new_tokens is not None
                else None
            ),
            *params,
        )
        if additive is not None:
            additive = self._to_batch(additive, 0.0)
        if scaling is not None:
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

    def apply_verify(
        self,
        logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        next_child: Optional[torch.Tensor] = None,
        next_sibling: Optional[torch.Tensor] = None,
    ) -> None:
        """Apply each verify row's own penalties to `logits` in place.

        A row's history adds the tokens from the root (exclusive; the history
        already counts it) to its node (inclusive), along the tree, or along the
        columns when `next_child` and `next_sibling` are None.
        """
        tables = self.pool.live_tables()
        if tables is None:
            # Every request of the step has exited.
            return
        params = self._params(tables)
        if _has_verify_kernel and logits.is_cuda:
            from sglang.kernels.ops.sampling.verify_penalties import (
                apply_verify_penalties,
            )

            apply_verify_penalties(
                logits,
                draft_tokens,
                next_child,
                next_sibling,
                self.batch_indices,
                self.penalty_indices,
                tables.counts,
                tables.lens,
                *params,
            )
        else:
            self._apply_verify_torch(
                logits,
                draft_tokens,
                next_child,
                next_sibling,
                tables,
                params.gather(self.penalty_indices),
            )

    def _apply_verify_torch(
        self,
        logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        next_child: Optional[torch.Tensor],
        next_sibling: Optional[torch.Tensor],
        tables: PenaltyTables,
        params: _StepParams,
    ) -> None:
        vocab_size = self.pool.vocab_size
        num_requests = self.penalty_indices.numel()
        width = draft_tokens.shape[1]
        device = logits.device
        tokens = draft_tokens[self.batch_indices].long()
        if next_child is None:
            parents = (torch.arange(width, device=device) - 1).expand_as(tokens)
        else:
            parents = _tree_parents(
                next_child[self.batch_indices], next_sibling[self.batch_indices]
            )
        in_vocab = (tokens >= 0) & (tokens < vocab_size)
        tokens = tokens.clamp(0, vocab_size - 1)

        # Every row walks up its path one node per step; the root has no parent
        # and never counts.
        row_counts = tables.counts[self.penalty_indices].repeat_interleave(width, dim=0)
        row_lens = tables.lens[self.penalty_indices].repeat_interleave(width)
        rows = torch.arange(num_requests * width, device=device)
        node = torch.arange(width, device=device).repeat(num_requests, 1)
        for _ in range(width):
            parent = parents.gather(1, node)
            counted = ((parent >= 0) & in_vocab.gather(1, node)).view(-1)
            row_counts.index_put_(
                (rows, tokens.gather(1, node).view(-1)),
                counted.to(torch.int32),
                accumulate=True,
            )
            row_lens += counted
            node = torch.where(parent >= 0, parent, node)

        additive, scaling = _compute_logit_penalties(
            row_counts,
            row_lens,
            *(
                None if values is None else values.repeat_interleave(width, dim=0)
                for values in params
            ),
        )
        row_indices = (
            self.batch_indices[:, None] * width + torch.arange(width, device=device)
        ).view(-1)
        penalized = logits[row_indices]
        if additive is not None:
            penalized.add_(additive)
        if scaling is not None:
            apply_scaling_penalties(penalized, scaling)
        logits[row_indices] = penalized

    def commit(
        self, token_ids: torch.Tensor, accept_mask: Optional[torch.Tensor] = None
    ) -> None:
        """token_ids: [batch_size] sampled ids, or [batch_size, k] verify tokens
        with `accept_mask` marking the ones that count."""
        token_ids = token_ids.view(self.batch_size, -1)
        if accept_mask is not None:
            accept_mask = accept_mask.view(self.batch_size, -1)
        self.pool.accumulate(
            self.batch_indices, self.penalty_indices, token_ids, accept_mask
        )


def _compute_logit_penalties(
    counts: torch.Tensor,
    lens: Optional[torch.Tensor],
    frequency: Optional[torch.Tensor],
    presence: Optional[torch.Tensor],
    repetition: Optional[torch.Tensor],
    min_new_tokens: Optional[torch.Tensor],
    stop_token_ids: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """(additive, scaling) [m, vocab] penalties of m histories, each None if unused."""
    vocab_size = counts.shape[1]
    device = counts.device
    present = counts > 0 if presence is not None or repetition is not None else None

    additive = None
    if frequency is not None or presence is not None or min_new_tokens is not None:
        additive = torch.zeros(counts.shape, dtype=torch.float32, device=device)
        if frequency is not None:
            additive.addcmul_(counts, frequency, value=-1)
        if presence is not None:
            mask = present.to(additive.dtype) if _is_npu else present
            # aclnnInplaceAddcmul rejects DT_BOOL tensors; promote on NPU only.
            additive.addcmul_(mask, presence, value=-1)
        if min_new_tokens is not None:
            stop_penalties = torch.zeros(
                (counts.shape[0], vocab_size + 1), dtype=torch.float32, device=device
            ).scatter_(1, stop_token_ids, float("-inf"))[:, :vocab_size]
            # torch.where, not a boolean index: no host sync, no -inf * 0.
            additive.add_(
                torch.where(lens.unsqueeze(1) < min_new_tokens, stop_penalties, 0.0)
            )

    scaling = None
    if repetition is not None:
        scaling = torch.where(present, repetition, 1.0)
    return additive, scaling


def _tree_parents(next_child: torch.Tensor, next_sibling: torch.Tensor) -> torch.Tensor:
    """[n, width] parent of each tree node, -1 at the root and off the tree."""
    num_trees, width = next_child.shape
    device = next_child.device

    def links(targets: torch.Tensor) -> torch.Tensor:
        # Column `width` absorbs the writes of missing links.
        return torch.where((targets >= 0) & (targets < width), targets, width)

    parents = torch.full((num_trees, width + 1), -1, dtype=torch.int64, device=device)
    nodes = torch.arange(width, device=device).expand(num_trees, width)
    parents.scatter_(1, links(next_child), nodes)
    # A node's next sibling shares its parent; k siblings need k - 1 steps.
    siblings = links(next_sibling)
    for _ in range(width - 1):
        parents.scatter_(1, siblings, parents[:, :width].clone())
    return parents[:, :width]


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
