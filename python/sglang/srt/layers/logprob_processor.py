from __future__ import annotations

import dataclasses
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

logger = logging.getLogger(__name__)


class LogprobStage(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclasses.dataclass
class LogprobResult:
    """Logprob fields produced by Input/OutputLogprobProcessor.

    Input (prefill) always fills token_logprobs; output (decode / scoring)
    fills on demand. write_input_to / write_output_to flush populated fields
    onto LogitsProcessorOutput, so the IPC / D2H wire format stays unchanged.
    """

    token_logprobs: Optional[torch.Tensor] = None
    top_logprobs_val: Optional[List] = None
    top_logprobs_idx: Optional[List] = None
    token_ids_logprobs_val: Optional[List] = None
    token_ids_logprobs_idx: Optional[List] = None

    def write_input_to(self, logits_output: LogitsProcessorOutput) -> None:
        if self.token_logprobs is not None:
            logits_output.input_token_logprobs = self.token_logprobs
        if self.top_logprobs_val is not None:
            logits_output.input_top_logprobs_val = self.top_logprobs_val
            logits_output.input_top_logprobs_idx = self.top_logprobs_idx
        if self.token_ids_logprobs_val is not None:
            logits_output.input_token_ids_logprobs_val = self.token_ids_logprobs_val
            logits_output.input_token_ids_logprobs_idx = self.token_ids_logprobs_idx

    def write_output_to(self, logits_output: LogitsProcessorOutput) -> None:
        if self.token_logprobs is not None:
            logits_output.next_token_logprobs = self.token_logprobs
        if self.top_logprobs_val is not None:
            logits_output.next_token_top_logprobs_val = self.top_logprobs_val
            logits_output.next_token_top_logprobs_idx = self.top_logprobs_idx
        if self.token_ids_logprobs_val is not None:
            logits_output.next_token_token_ids_logprobs_val = (
                self.token_ids_logprobs_val
            )
            logits_output.next_token_token_ids_logprobs_idx = (
                self.token_ids_logprobs_idx
            )


@dataclasses.dataclass(frozen=True)
class DistributedLogprobContext:
    """TP-sharded input-logprob callbacks and contiguous vocab ownership."""

    tp_group: Any
    vocab_start_index: int
    vocab_end_index: int
    vocab_size: int
    get_local_logits_fn: Callable
    gather_sampled_logits_fn: Callable


@dataclasses.dataclass(frozen=True)
class _TokenIdsChunkEntry:
    token_ids: Optional[List[int]]
    num_rows: int
    continue_previous: bool


@dataclasses.dataclass(frozen=True)
class _TokenIdsChunkPlan:
    row_indices: torch.Tensor
    token_ids: torch.Tensor
    entries: List[_TokenIdsChunkEntry]
    next_split_pruned_len: int


def compute_row_log_normalizer(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row ``(max, logsumexp - max)`` in fp32.

    Consumers compute ``logprob[i] = (logit[i] - max) - log_sum``, the same
    shift-invariant order as log-softmax; a single absolute normalizer would
    round the log_sum term away for rows with a large common offset.
    """
    if logits.is_cuda:
        from sglang.srt.layers.logsumexp import row_logsumexp

        return row_logsumexp(logits)
    x = logits.float()
    row_max = x.amax(dim=-1)
    row_log_sum = torch.logsumexp(x - row_max[:, None], dim=-1)
    row_log_sum = torch.where(row_max.isinf(), 0.0, row_log_sum)
    return row_max, row_log_sum


def compute_distributed_row_log_normalizer(
    local_logits: torch.Tensor,
    valid_vocab_size: int,
    tp_group: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a global row normalizer without gathering vocab logits.

    ``local_logits`` follows rank-local contiguous vocab order. Columns after
    ``valid_vocab_size`` are padding and do not participate. The exchanged
    tensors are one fp32 max and one fp32 exponential sum per row.
    """
    if not 0 <= valid_vocab_size <= local_logits.shape[1]:
        raise ValueError(
            f"valid_vocab_size={valid_vocab_size} is incompatible with "
            f"local logits width {local_logits.shape[1]}"
        )

    if valid_vocab_size == 0:
        local_max = torch.full(
            (local_logits.shape[0],),
            float("-inf"),
            dtype=torch.float32,
            device=local_logits.device,
        )
        local_log_sum = torch.zeros_like(local_max)
    else:
        valid_logits = local_logits[:, :valid_vocab_size]
        local_max, local_log_sum = compute_row_log_normalizer(valid_logits)

    global_max = local_max.clone()
    torch.distributed.all_reduce(
        global_max,
        op=torch.distributed.ReduceOp.MAX,
        group=tp_group.device_group,
    )

    if valid_vocab_size == 0:
        local_exp_sum = torch.zeros_like(global_max)
    else:
        local_exp_sum = torch.exp((local_max - global_max) + local_log_sum)
    global_exp_sum = tp_group.all_reduce(local_exp_sum)
    return global_max, torch.log(global_exp_sum)


def get_distributed_token_scores(
    local_logits: torch.Tensor,
    row_indices: torch.Tensor,
    token_ids: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: Any,
) -> torch.Tensor:
    """Look up raw token scores from contiguous TP vocab shards.

    Exactly one rank contributes each requested token score; all other ranks
    contribute zero. A SUM reduction therefore returns the selected score on
    every rank while exchanging only one fp32 value per request.
    """
    if row_indices.ndim != 1 or token_ids.ndim != 1:
        raise ValueError("row_indices and token_ids must be one-dimensional")
    if row_indices.shape != token_ids.shape:
        raise ValueError("row_indices and token_ids must have the same shape")
    if row_indices.numel() == 0:
        return local_logits.new_empty(0, dtype=torch.float32)

    owned = (token_ids >= vocab_start_index) & (token_ids < vocab_end_index)
    local_width = local_logits.shape[1]
    if local_width == 0:
        local_scores = local_logits.new_zeros(token_ids.shape, dtype=torch.float32)
    else:
        local_indices = (token_ids - vocab_start_index).clamp(
            min=0, max=local_width - 1
        )
        selected = local_logits[row_indices, local_indices].float()
        local_scores = torch.where(owned, selected, torch.zeros_like(selected))
    return tp_group.all_reduce(local_scores)


def get_distributed_topk(
    local_logits: torch.Tensor,
    valid_vocab_size: int,
    max_k: int,
    vocab_start_index: int,
    vocab_size: int,
    tp_group: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return global top-k raw logits from contiguous TP vocab shards.

    Each rank first selects at most ``max_k`` of its *valid* local columns.
    Values and absolute token ids are then all-gathered and reduced locally.
    This exchanges ``O(rows * tp_size * max_k)`` candidates instead of a
    full-vocabulary tensor.  Padding columns are represented by ``-inf`` and
    an out-of-vocabulary sentinel, so they cannot win the final top-k.

    Ties retain the backend's ``torch.topk`` behavior.  In particular, equal
    logits that straddle shards are valid top-k members but their relative
    order is not promised to match a monolithic full-vocab ``topk``.
    """
    if max_k < 0:
        raise ValueError(f"max_k must be non-negative, got {max_k}")
    if not 0 <= valid_vocab_size <= local_logits.shape[1]:
        raise ValueError(
            f"valid_vocab_size={valid_vocab_size} is incompatible with "
            f"local logits width {local_logits.shape[1]}"
        )

    rows = local_logits.shape[0]
    if max_k == 0:
        return (
            local_logits.new_empty((rows, 0), dtype=torch.float32),
            torch.empty((rows, 0), dtype=torch.long, device=local_logits.device),
        )

    local_values = torch.full(
        (rows, max_k),
        float("-inf"),
        dtype=torch.float32,
        device=local_logits.device,
    )
    local_indices = torch.full(
        (rows, max_k),
        vocab_size,
        dtype=torch.long,
        device=local_logits.device,
    )
    local_k = min(max_k, valid_vocab_size)
    if local_k:
        values, indices = local_logits[:, :valid_vocab_size].topk(local_k, dim=-1)
        local_values[:, :local_k] = values.float()
        local_indices[:, :local_k] = indices + vocab_start_index

    candidate_values = tp_group.all_gather(local_values, dim=1)
    candidate_indices = tp_group.all_gather(local_indices, dim=1)
    values, candidate_positions = candidate_values.topk(max_k, dim=-1)
    indices = candidate_indices.gather(1, candidate_positions)
    return values, indices


def append_distributed_topk_chunk(
    values: torch.Tensor,
    indices: torch.Tensor,
    top_k_nums: List[int],
    pruned_lens: List[int],
    top_logprobs_val: List,
    top_logprobs_idx: List,
    split_pruned_len: int,
) -> int:
    """Restore prompt top-k responses from row-major distributed results."""
    pt = 0
    next_split_pruned_len = 0
    for n, (k, original_pruned_len) in enumerate(zip(top_k_nums, pruned_lens)):
        current_split = split_pruned_len if n == 0 else 0
        pruned_len = original_pruned_len - current_split
        if pruned_len <= 0:
            top_logprobs_val.append([])
            top_logprobs_idx.append([])
            continue

        available_rows = min(pruned_len, max(values.shape[0] - pt, 0))
        val = values[pt : pt + available_rows, :k].tolist()
        idx = indices[pt : pt + available_rows, :k].tolist()
        if current_split:
            top_logprobs_val[-1].extend(val)
            top_logprobs_idx[-1].extend(idx)
        else:
            top_logprobs_val.append(val)
            top_logprobs_idx.append(idx)
        if available_rows < pruned_len:
            next_split_pruned_len = current_split + available_rows
        pt += pruned_len
    return next_split_pruned_len


def _build_token_ids_chunk_plan(
    token_ids_logprobs: List[Optional[List[int]]],
    pruned_lens: List[int],
    split_pruned_len: int,
    num_rows: int,
    device: torch.device,
) -> _TokenIdsChunkPlan:
    """Flatten ragged explicit-token probes into one collective request."""
    request_rows: List[int] = []
    request_token_ids: List[int] = []
    entries: List[_TokenIdsChunkEntry] = []
    pt = 0
    next_split_pruned_len = 0

    for n, (token_ids, original_pruned_len) in enumerate(
        zip(token_ids_logprobs, pruned_lens)
    ):
        current_split = split_pruned_len if n == 0 else 0
        pruned_len = original_pruned_len - current_split
        continue_previous = current_split > 0

        if pruned_len <= 0:
            entries.append(_TokenIdsChunkEntry(token_ids, 0, False))
            continue

        available_rows = min(pruned_len, max(num_rows - pt, 0))
        if available_rows < pruned_len:
            next_split_pruned_len = current_split + available_rows

        if token_ids:
            for row in range(pt, pt + available_rows):
                request_rows.extend([row] * len(token_ids))
                request_token_ids.extend(token_ids)

        entries.append(
            _TokenIdsChunkEntry(token_ids, available_rows, continue_previous)
        )
        pt += pruned_len

    return _TokenIdsChunkPlan(
        row_indices=torch.tensor(request_rows, dtype=torch.long, device=device),
        token_ids=torch.tensor(request_token_ids, dtype=torch.long, device=device),
        entries=entries,
        next_split_pruned_len=next_split_pruned_len,
    )


def _append_token_ids_chunk_from_flat_scores(
    plan: _TokenIdsChunkPlan,
    flat_scores: torch.Tensor,
    token_ids_logprobs_val: List,
    token_ids_logprobs_idx: List,
) -> None:
    """Restore the existing per-sequence/per-row response shape."""
    score_pt = 0
    for entry in plan.entries:
        token_ids = entry.token_ids
        width = len(token_ids) if token_ids else 0
        count = entry.num_rows * width
        if token_ids is None:
            val, idx = [], []
        elif width:
            values = flat_scores[score_pt : score_pt + count].reshape(
                entry.num_rows, width
            )
            val = values.tolist()
            idx = [token_ids for _ in range(entry.num_rows)]
        else:
            # [] requests a zero-width result for every prompt row; None is
            # the opt-out sentinel and returns no rows at all.
            val = [[] for _ in range(entry.num_rows)]
            idx = [[] for _ in range(entry.num_rows)]
        score_pt += count

        if entry.continue_previous:
            token_ids_logprobs_val[-1].extend(val)
            token_ids_logprobs_idx[-1].extend(idx)
        else:
            token_ids_logprobs_val.append(val)
            token_ids_logprobs_idx.append(idx)

    assert score_pt == flat_scores.numel()


def get_top_logprobs_raw(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    no_copy_to_cpu: bool = False,
):
    max_k = max(top_logprobs_nums)
    values, indices = logprobs.topk(max_k, dim=-1)
    if not no_copy_to_cpu:
        values = values.tolist()
        indices = indices.tolist()

    top_logprobs_val = []
    top_logprobs_idx = []

    if stage == LogprobStage.DECODE:
        for i, k in enumerate(top_logprobs_nums):
            top_logprobs_val.append(values[i][:k])
            top_logprobs_idx.append(indices[i][:k])
    else:
        pt = 0
        for k, pruned_len in zip(top_logprobs_nums, extend_logprob_pruned_lens_cpu):
            if pruned_len <= 0:
                top_logprobs_val.append([])
                top_logprobs_idx.append([])
                continue

            top_logprobs_val.append([values[pt + j][:k] for j in range(pruned_len)])
            top_logprobs_idx.append([indices[pt + j][:k] for j in range(pruned_len)])
            pt += pruned_len

    return top_logprobs_val, top_logprobs_idx


def get_top_logprobs(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    no_copy_to_cpu: bool = False,
):
    return get_top_logprobs_raw(
        logprobs,
        top_logprobs_nums,
        stage=LogprobStage.DECODE,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_token_ids_logprobs_raw(
    logprobs: torch.Tensor,
    token_ids_logprobs_list: List[Optional[List[int]]],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    no_copy_to_cpu: bool = False,
):
    vals, idxs = [], []
    if stage == LogprobStage.DECODE:
        for i, token_ids in enumerate(token_ids_logprobs_list):
            if token_ids is None:
                vals.append([])
                idxs.append([])
            else:
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(
                    logprobs.device, non_blocking=True
                )
                row = logprobs[i, token_ids_tensor]
                vals.append(row if no_copy_to_cpu else row.tolist())
                idxs.append(token_ids)
    else:  # prefill
        pt = 0
        for i, (token_ids, pruned_len) in enumerate(
            zip(token_ids_logprobs_list, extend_logprob_pruned_lens_cpu)
        ):
            if pruned_len <= 0:
                vals.append([])
                idxs.append([])
                continue
            if token_ids is None:
                # The sequence's rows still occupy logprobs; step over them.
                vals.append([])
                idxs.append([])
                pt += pruned_len
                continue
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(
                logprobs.device, non_blocking=True
            )
            pos_logprobs = logprobs[pt : pt + pruned_len, token_ids_tensor]
            vals.append(pos_logprobs if no_copy_to_cpu else pos_logprobs.tolist())
            idxs.append([token_ids for _ in range(pruned_len)])
            pt += pruned_len
    return vals, idxs


def get_token_ids_logprobs(logprobs, token_ids_logprobs, no_copy_to_cpu=False):
    return get_token_ids_logprobs_raw(
        logprobs,
        token_ids_logprobs,
        stage=LogprobStage.DECODE,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_top_logprobs_chunk(
    logprobs: torch.Tensor,
    top_k_nums: List[int],
    pruned_lens: List[int],
    top_logprobs_val: List,
    top_logprobs_idx: List,
    split_pruned_len: int,
    log_normalizer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    precomputed_topk: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> int:
    """Get top-k logprobs for each sequence in the chunk.

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size].
            With ``log_normalizer`` set, raw logits instead; top-k runs on the
            logits (same order) and values are normalized by subtraction.
        top_k_nums: List of top-k numbers for each sequence
        pruned_lens: List of pruned lengths for each sequence
        top_logprobs_val: List to store top-k logprob values
        top_logprobs_idx: List to store top-k token indices
        split_pruned_len: Length of pruned tokens from previous chunk
        log_normalizer: Per-row (max, logsumexp - max) of the logits
        precomputed_topk: (raw fp32 top values, indices) from the fused
            logsumexp+top-k kernel, sorted with lowest-index tie-breaking

    Returns:
        int: Number of remaining tokens to process in next chunk
    """
    # Empty chunks still walk the slice to emit placeholder entries.
    max_k = max(top_k_nums)
    if log_normalizer is not None:
        row_max, row_log_sum = log_normalizer
        if precomputed_topk is not None:
            values_tensor, indices_tensor = precomputed_topk
        else:
            values_tensor, indices_tensor = logprobs.topk(max_k, dim=1)
        values_tensor = (values_tensor.float() - row_max[:, None]) - row_log_sum[
            :, None
        ]
    else:
        values_tensor, indices_tensor = logprobs.topk(max_k, dim=1)
    values = values_tensor.tolist()
    indices = indices_tensor.tolist()

    pt = 0
    next_split_pruned_len = 0
    for n, (k, pruned_len) in enumerate(zip(top_k_nums, pruned_lens)):
        if n == 0:
            # For the first sequence, adjust the pruned length
            pruned_len -= split_pruned_len
        else:
            # After the first sequence, no split in the middle
            split_pruned_len = 0

        if pruned_len <= 0:
            # if pruned length is less than or equal to 0,
            # there is no top-k logprobs to process
            top_logprobs_val.append([])
            top_logprobs_idx.append([])
            continue

        # Get the top-k logprobs
        val = []
        idx = []
        for j in range(pruned_len):
            # Handle remaining tokens in next chunk if any
            if pt + j >= len(values):
                next_split_pruned_len = split_pruned_len + j
                break
            # Append the top-k logprobs
            val.append(values[pt + j][:k])
            idx.append(indices[pt + j][:k])

        # Append or extend based on whether the sequence was split across chunks
        # Split-sequence continuations extend; everyone else owns a fresh
        # (possibly empty) entry.
        if split_pruned_len > 0:
            top_logprobs_val[-1].extend(val)
            top_logprobs_idx[-1].extend(idx)
        else:
            top_logprobs_val.append(val)
            top_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def get_token_ids_logprobs_chunk(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[int],
    pruned_lens: List[int],
    token_ids_logprobs_val: List,
    token_ids_logprobs_idx: List,
    split_pruned_len: int = 0,
    log_normalizer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """Get token_ids logprobs for each sequence in the chunk.

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size].
            With ``log_normalizer`` set, raw logits instead; gathered rows are
            normalized by subtraction.
        token_ids_logprobs: List of token IDs for each sequence
        pruned_lens: List of pruned lengths for each sequence
        token_ids_logprobs_val: List to store token logprob values
        token_ids_logprobs_idx: List to store token indices
        split_pruned_len: Length of pruned tokens from previous chunk
        log_normalizer: Per-row (max, logsumexp - max) of the logits

    Returns:
        int: Number of remaining tokens to process in next chunk
    """
    # Empty chunks still walk the slice to emit placeholder entries.
    if log_normalizer is not None:
        row_max, row_log_sum = log_normalizer
    pt = 0
    next_split_pruned_len = 0
    for n, (token_ids, pruned_len) in enumerate(
        zip(
            token_ids_logprobs,
            pruned_lens,
        )
    ):
        # Adjust pruned length for first sequence
        if n == 0:
            pruned_len -= split_pruned_len
        else:
            split_pruned_len = 0

        if pruned_len <= 0:
            # if pruned length is less than or equal to 0,
            # there is no token ids logprobs to process
            token_ids_logprobs_val.append([])
            token_ids_logprobs_idx.append([])
            continue

        # Get the token ids logprobs
        val = []
        idx = []
        for j in range(pruned_len):
            # Handle remaining tokens in next chunk if any
            if pt + j >= logprobs.shape[0]:
                next_split_pruned_len = split_pruned_len + j
                break
            if token_ids is not None:
                row = logprobs[pt + j, token_ids]
                if log_normalizer is not None:
                    row = (row.float() - row_max[pt + j]) - row_log_sum[pt + j]
                val.append(row.tolist())
                idx.append(token_ids)

        # Split-sequence continuations extend; everyone else owns a fresh
        # (possibly empty) entry.
        if split_pruned_len > 0:
            token_ids_logprobs_val[-1].extend(val)
            token_ids_logprobs_idx[-1].extend(idx)
        else:
            token_ids_logprobs_val.append(val)
            token_ids_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def compute_spec_v2_logprobs(
    batch,
    logits_output,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    speculative_num_steps: int,
):
    """Compute logprobs for accepted tokens after spec v2 verify sampling.

    Gathers logits at accepted positions, applies log_softmax (temperature-scaled
    if not greedy), and populates logits_output.next_token_logprobs (plus optional
    top-k / token-ids logprobs) so they flow through copy_to_cpu().
    """
    bs = len(batch.seq_lens)
    max_accept = speculative_num_steps + 1
    device = predict.device

    flat_accept_idx = accept_index.long().reshape(-1)
    gathered_logits = logits_output.next_token_logits[flat_accept_idx]

    if batch.sampling_info.is_all_greedy or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
        gathered_logprobs = torch.nn.functional.log_softmax(gathered_logits, dim=-1)
    else:
        temperatures = torch.repeat_interleave(
            batch.sampling_info.temperatures,
            max_accept,
            dim=0,
        )
        gathered_logprobs = torch.nn.functional.log_softmax(
            gathered_logits / temperatures, dim=-1
        )
    gathered_logprobs.clamp_(min=torch.finfo(gathered_logprobs.dtype).min)

    accepted_token_ids = predict[flat_accept_idx]
    token_logprobs = gathered_logprobs[
        torch.arange(bs * max_accept, device=device),
        accepted_token_ids.long(),
    ]
    logits_output.next_token_logprobs = token_logprobs.reshape(bs, max_accept)

    if batch.top_logprobs_nums and any(x > 0 for x in batch.top_logprobs_nums):
        top_logprobs_nums_expanded = [
            num for num in batch.top_logprobs_nums for _ in range(max_accept)
        ]
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(
            gathered_logprobs, top_logprobs_nums_expanded, no_copy_to_cpu=True
        )

    if batch.token_ids_logprobs and any(
        x is not None for x in batch.token_ids_logprobs
    ):
        token_ids_logprobs_expanded = [
            ids for ids in batch.token_ids_logprobs for _ in range(max_accept)
        ]
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(
            gathered_logprobs, token_ids_logprobs_expanded, no_copy_to_cpu=True
        )


def _deterministic_inference_enabled() -> bool:
    """True when serving with --enable-deterministic-inference.

    Fails open: bare constructions (unit tests) have no published config
    namespaces, and plain serving is the not-deterministic case.
    """
    try:
        return bool(get_exec().deterministic.enable_deterministic_inference)
    except ValueError:
        return False


class InputLogprobProcessor:
    """Input (prefill) logprob processing: single-pass or chunked.

    Logits are computed through the injected ``get_logits_fn(hidden_states,
    lm_head, logits_metadata)`` callable, so this class stays decoupled from
    the lm_head / TP-gather machinery in LogitsProcessor.
    """

    def __init__(self):
        # enable chunked logprobs processing
        self.enable_logprobs_chunk = envs.SGLANG_ENABLE_LOGPROB_CHUNK.get()
        # chunk size for logprobs processing
        self.logprobs_chunk_size = envs.SGLANG_LOGPROB_CHUNK_SIZE.get()
        # Compute input logprobs from logits + logsumexp, skipping the
        # full-vocab log-softmax materialization. Deterministic inference
        # keeps the exact log_softmax path: the fused logsumexp reduces in a
        # different order, which breaks the prefill/decode logprob
        # bit-identity that mode guarantees.
        self.enable_fast_input_logprobs = (
            envs.SGLANG_ENABLE_FAST_INPUT_LOGPROBS.get()
            and not _deterministic_inference_enabled()
        )
        self.enable_distributed_input_logprobs = (
            envs.SGLANG_ENABLE_DISTRIBUTED_INPUT_LOGPROBS.get()
        )

    def forward(
        self,
        pruned_states: torch.Tensor,
        sample_indices: Optional[torch.Tensor],
        input_logprob_indices: torch.Tensor,
        token_to_seq_idx: list[int],
        lm_head: VocabParallelEmbedding,
        get_logits_fn: Callable,
        logits_metadata: LogitsMetadata,
        skip_chunking_for_dp_attn: bool = False,
        distributed_context: Optional[DistributedLogprobContext] = None,
    ) -> Tuple[LogprobResult, torch.Tensor]:
        # Non-chunked = one chunk covering every row. DP-attention must stay
        # single-chunk: the collective schedule cannot depend on per-rank rows.
        if (
            not self.enable_logprobs_chunk
            or pruned_states.shape[0] <= self.logprobs_chunk_size
            or skip_chunking_for_dp_attn
        ):
            chunk_size = max(pruned_states.shape[0], 1)
        else:
            chunk_size = self.logprobs_chunk_size

        return self._forward_by_chunk(
            pruned_states,
            sample_indices,
            input_logprob_indices,
            token_to_seq_idx,
            lm_head,
            get_logits_fn,
            logits_metadata,
            chunk_size,
            distributed_context,
        )

    def _forward_by_chunk(
        self,
        pruned_states: torch.Tensor,
        sample_indices: torch.Tensor,
        input_logprob_indices: torch.Tensor,
        token_to_seq_idx: list[int],
        lm_head: VocabParallelEmbedding,
        get_logits_fn: Callable,
        logits_metadata: LogitsMetadata,
        chunk_size: int,
        distributed_context: Optional[DistributedLogprobContext] = None,
    ) -> Tuple[LogprobResult, torch.Tensor]:
        """Compute input logprobs chunk by chunk to cap peak memory."""
        total_size = pruned_states.shape[0]
        num_chunks = (total_size + chunk_size - 1) // chunk_size

        token_logprobs = []
        if logits_metadata.extend_return_top_logprob:
            top_logprobs_val = []
            top_logprobs_idx = []
        else:
            top_logprobs_val = None
            top_logprobs_idx = None
        if logits_metadata.extend_token_ids_logprob:
            token_ids_logprobs_val = []
            token_ids_logprobs_idx = []
        else:
            token_ids_logprobs_val = None
            token_ids_logprobs_idx = None

        # If a single sequence is split into multiple chunks, we need to keep track
        # of the pruned length of the sequences in the previous chunks.
        split_len_topk = 0
        split_len_token_ids = 0

        fused_kernel, fused_max_k = None, 0
        if self.enable_fast_input_logprobs and pruned_states.is_cuda:
            from sglang.srt.layers.logsumexp import (
                FUSED_TOPK_MAX_K,
                row_logsumexp_topk,
            )

            fused_kernel, fused_max_k = row_logsumexp_topk, FUSED_TOPK_MAX_K

        use_distributed_logprobs = (
            distributed_context is not None and self.enable_fast_input_logprobs
        )

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_size)

            # Notify lm_head LoRA about the current chunk so it can swap
            # to the precomputed per-chunk batch_info.  This is a no-op
            # for non-LoRA lm_head modules.
            if num_chunks > 1 and hasattr(lm_head, "set_lm_head_pass"):
                lm_head.set_lm_head_pass(i)

            # Get indices for this chunk
            chunk_mask = (input_logprob_indices >= start_idx) & (
                input_logprob_indices < end_idx
            )
            global_indices = input_logprob_indices[chunk_mask]
            chunk_indices = global_indices - start_idx
            # Get the positions in the original array where chunk_mask is True
            # This is needed to correctly index into extend_input_logprob_token_ids_gpu
            mask_indices = torch.nonzero(chunk_mask, as_tuple=True)[0]

            # Get the logits for this chunk. Each chunk must own its output:
            # writing through the shared graph logits buffer would alias
            # chunks whose shape happens to match the buffer.
            chunk_states = pruned_states[start_idx:end_idx]
            if use_distributed_logprobs:
                chunk_logits = distributed_context.get_local_logits_fn(
                    chunk_states, lm_head, logits_metadata
                )
            else:
                chunk_logits = get_logits_fn(
                    chunk_states,
                    lm_head,
                    logits_metadata,
                    use_logits_buffer=num_chunks == 1,
                )

            # Initialize sampled_logits on first chunk
            if i == 0:
                sampled_vocab_size = (
                    distributed_context.vocab_size
                    if use_distributed_logprobs
                    else chunk_logits.shape[1]
                )
                sampled_logits = torch.empty(
                    (sample_indices.shape[0], sampled_vocab_size),
                    dtype=chunk_logits.dtype,
                    device=chunk_logits.device,
                )

            # Handle sampled logits for the chunk if needed
            # This must be done before the continue statement to ensure all sampled_logits are filled
            chunk_sample_mask = (sample_indices >= start_idx) & (
                sample_indices < end_idx
            )
            if chunk_sample_mask.any():
                chunk_sample_indices = sample_indices[chunk_sample_mask] - start_idx
                chunk_sampled_logits = chunk_logits[chunk_sample_indices]
                if use_distributed_logprobs:
                    chunk_sampled_logits = distributed_context.gather_sampled_logits_fn(
                        chunk_sampled_logits
                    )
                sampled_logits[chunk_sample_mask] = chunk_sampled_logits

            # Zero-logprob-row chunks still need the per-sequence bookkeeping below.
            chunk_logprobs = chunk_logits[chunk_indices]
            del chunk_logits

            # End at the last row inside the chunk; token_to_seq_idx[end_idx]
            # belongs to the next chunk and would emit its sequence twice.
            chunk_slice = slice(
                token_to_seq_idx[start_idx], token_to_seq_idx[end_idx - 1] + 1
            )

            if use_distributed_logprobs:
                valid_vocab_size = (
                    distributed_context.vocab_end_index
                    - distributed_context.vocab_start_index
                )
                if chunk_logprobs.shape[0] > 0:
                    row_max, row_log_sum = compute_distributed_row_log_normalizer(
                        chunk_logprobs,
                        valid_vocab_size,
                        distributed_context.tp_group,
                    )

                    target_token_ids = (
                        logits_metadata.extend_input_logprob_token_ids_gpu[mask_indices]
                    ).long()
                    target_rows = torch.arange(
                        chunk_logprobs.shape[0],
                        device=chunk_logprobs.device,
                        dtype=torch.long,
                    )

                    explicit_plan = None
                    lookup_rows = target_rows
                    lookup_token_ids = target_token_ids
                    if logits_metadata.extend_token_ids_logprob:
                        explicit_plan = _build_token_ids_chunk_plan(
                            logits_metadata.token_ids_logprobs[chunk_slice],
                            logits_metadata.extend_logprob_pruned_lens_cpu[chunk_slice],
                            split_len_token_ids,
                            chunk_logprobs.shape[0],
                            chunk_logprobs.device,
                        )
                        lookup_rows = torch.cat(
                            (lookup_rows, explicit_plan.row_indices)
                        )
                        lookup_token_ids = torch.cat(
                            (lookup_token_ids, explicit_plan.token_ids)
                        )

                    raw_scores = get_distributed_token_scores(
                        chunk_logprobs,
                        lookup_rows,
                        lookup_token_ids,
                        distributed_context.vocab_start_index,
                        distributed_context.vocab_end_index,
                        distributed_context.tp_group,
                    )
                    normalized_scores = (
                        raw_scores - row_max[lookup_rows]
                    ) - row_log_sum[lookup_rows]
                    num_targets = target_rows.numel()
                    token_logprobs.append(normalized_scores[:num_targets])

                    if logits_metadata.extend_return_top_logprob:
                        top_k_nums = logits_metadata.top_logprobs_nums[chunk_slice]
                        top_values, top_indices = get_distributed_topk(
                            chunk_logprobs,
                            valid_vocab_size,
                            max(top_k_nums),
                            distributed_context.vocab_start_index,
                            distributed_context.vocab_size,
                            distributed_context.tp_group,
                        )
                        top_values = (top_values - row_max[:, None]) - row_log_sum[
                            :, None
                        ]
                        split_len_topk = append_distributed_topk_chunk(
                            top_values,
                            top_indices,
                            top_k_nums,
                            logits_metadata.extend_logprob_pruned_lens_cpu[chunk_slice],
                            top_logprobs_val,
                            top_logprobs_idx,
                            split_len_topk,
                        )

                    if explicit_plan is not None:
                        _append_token_ids_chunk_from_flat_scores(
                            explicit_plan,
                            normalized_scores[num_targets:],
                            token_ids_logprobs_val,
                            token_ids_logprobs_idx,
                        )
                        split_len_token_ids = explicit_plan.next_split_pruned_len
                else:
                    token_logprobs.append(
                        chunk_logprobs.new_empty(0, dtype=torch.float32)
                    )
                    if logits_metadata.extend_return_top_logprob:
                        top_k_nums = logits_metadata.top_logprobs_nums[chunk_slice]
                        empty_values = chunk_logprobs.new_empty(
                            (0, max(top_k_nums)), dtype=torch.float32
                        )
                        empty_indices = torch.empty(
                            (0, max(top_k_nums)),
                            dtype=torch.long,
                            device=chunk_logprobs.device,
                        )
                        split_len_topk = append_distributed_topk_chunk(
                            empty_values,
                            empty_indices,
                            top_k_nums,
                            logits_metadata.extend_logprob_pruned_lens_cpu[chunk_slice],
                            top_logprobs_val,
                            top_logprobs_idx,
                            split_len_topk,
                        )
                    if logits_metadata.extend_token_ids_logprob:
                        explicit_plan = _build_token_ids_chunk_plan(
                            logits_metadata.token_ids_logprobs[chunk_slice],
                            logits_metadata.extend_logprob_pruned_lens_cpu[chunk_slice],
                            split_len_token_ids,
                            0,
                            chunk_logprobs.device,
                        )
                        _append_token_ids_chunk_from_flat_scores(
                            explicit_plan,
                            chunk_logprobs.new_empty(0, dtype=torch.float32),
                            token_ids_logprobs_val,
                            token_ids_logprobs_idx,
                        )
                        split_len_token_ids = explicit_plan.next_split_pruned_len
                del chunk_logprobs
                continue

            chunk_precomputed_topk = None
            if self.enable_fast_input_logprobs:
                # Every consumer below needs only small gathers / top-k plus a
                # per-row normalizer, so keep the raw logits and skip the
                # full-vocab log-softmax materialization entirely. When top-k
                # is requested, the fused kernel produces the normalizer and
                # the top-k in the same single read of the logits.
                max_k = (
                    max(logits_metadata.top_logprobs_nums[chunk_slice])
                    if logits_metadata.extend_return_top_logprob
                    else 0
                )
                if 0 < max_k <= fused_max_k:
                    row_max, row_log_sum, top_vals, top_idx = fused_kernel(
                        chunk_logprobs, max_k
                    )
                    chunk_log_normalizer = (row_max, row_log_sum)
                    chunk_precomputed_topk = (top_vals, top_idx)
                else:
                    chunk_log_normalizer = compute_row_log_normalizer(chunk_logprobs)
            else:
                # Free the raw logits before the out-of-place log_softmax:
                # keeping all three alive is a 3x peak, which OOMs when the
                # single chunk covers a large batch.
                chunk_log_normalizer = None
                chunk_logprobs = torch.nn.functional.log_softmax(chunk_logprobs, dim=-1)

            # Get the logprob of top-k tokens
            if logits_metadata.extend_return_top_logprob:
                top_k_nums = logits_metadata.top_logprobs_nums[chunk_slice]
                pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu[
                    chunk_slice
                ]
                split_len_topk = get_top_logprobs_chunk(
                    chunk_logprobs,
                    top_k_nums,
                    pruned_lens,
                    top_logprobs_val,
                    top_logprobs_idx,
                    split_len_topk,
                    log_normalizer=chunk_log_normalizer,
                    precomputed_topk=chunk_precomputed_topk,
                )

            # Get the logprob of given token id
            if logits_metadata.extend_token_ids_logprob:
                token_ids_logprobs = logits_metadata.token_ids_logprobs[chunk_slice]
                pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu[
                    chunk_slice
                ]
                split_len_token_ids = get_token_ids_logprobs_chunk(
                    chunk_logprobs,
                    token_ids_logprobs,
                    pruned_lens,
                    token_ids_logprobs_val,
                    token_ids_logprobs_idx,
                    split_len_token_ids,
                    log_normalizer=chunk_log_normalizer,
                )

            # Get the logprob of the requested token ids
            chunk_token_logprobs = chunk_logprobs[
                torch.arange(chunk_logprobs.shape[0], device=chunk_logprobs.device),
                logits_metadata.extend_input_logprob_token_ids_gpu[mask_indices],
            ]
            if chunk_log_normalizer is not None:
                row_max, row_log_sum = chunk_log_normalizer
                chunk_token_logprobs = (
                    chunk_token_logprobs.float() - row_max
                ) - row_log_sum
            token_logprobs.append(chunk_token_logprobs)
            # Free before the next chunk's logits (bf16 + fp32) materialize.
            del chunk_logprobs

        # Restore the full-pruned lm_head batch_info after chunk iteration.
        if num_chunks > 1 and hasattr(lm_head, "reset_lm_head_pass"):
            assert hasattr(
                lm_head, "set_lm_head_pass"
            ), "lm_head must have set_lm_head_pass method and reset_lm_head_pass method at the same time"
            lm_head.reset_lm_head_pass()

        # Concatenate the results
        token_logprobs = torch.cat(token_logprobs, dim=0)

        return (
            LogprobResult(
                token_logprobs=token_logprobs,
                top_logprobs_val=top_logprobs_val,
                top_logprobs_idx=top_logprobs_idx,
                token_ids_logprobs_val=token_ids_logprobs_val,
                token_ids_logprobs_idx=token_ids_logprobs_idx,
            ),
            sampled_logits,
        )


def get_token_ids_logprobs_batch_optimized(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[List[int]],
) -> Tuple[List, List]:
    """
    Vectorized batch processing for token ID logprobs extraction.

    Uses a single GPU kernel call for the entire batch instead of multiple
    separate calls, significantly improving performance for large batches.

    Args:
        logprobs: Log probabilities tensor [batch_size, vocab_size]
        token_ids_logprobs: List of token IDs to extract logprobs for

    Example:
        # Input: batch_size=3, vocab_size=5
        logprobs = torch.tensor([
            [-1.2, -2.1, -0.8, -3.0, -1.5],  # batch 0
            [-0.5, -1.8, -2.2, -1.1, -2.7],  # batch 1
            [-2.0, -0.9, -1.4, -2.8, -1.6],  # batch 2
        ])
        token_ids_logprobs = [[1, 3], [2], [0, 2, 4]]

        # Output:
        # values = [tensor([-2.1, -3.0]), tensor([-2.2]), tensor([-2.0, -1.4, -1.6])]
        # indices = [[1, 3], [2], [0, 2, 4]]
    """
    batch_size = len(token_ids_logprobs)
    device = logprobs.device

    # Step 1: Calculate lengths for each request, treating None as empty list
    # Example: [[1, 3], [2], [0, 2, 4]] -> token_lengths = tensor([2, 1, 3])
    token_lengths = torch.tensor(
        [len(token_ids or []) for token_ids in token_ids_logprobs], device=device
    )
    total_tokens = int(token_lengths.sum().item())  # 2 + 1 + 3 = 6

    # Handle edge case where no tokens are requested
    if total_tokens == 0:
        return [logprobs.new_empty(0) for _ in token_ids_logprobs], [
            [] for _ in token_ids_logprobs
        ]

    # Step 2: Build flattened indices using torch operations
    # Example: row_indices = [0, 0, 1, 2, 2, 2] (batch indices repeated by their lengths)
    row_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device), token_lengths
    )
    # Example: col_indices = [1, 3, 2, 0, 2, 4] (flattened token IDs from all requests)
    col_indices = torch.tensor(
        [
            token_id
            for token_ids in token_ids_logprobs
            for token_id in (token_ids or [])
        ],
        device=device,
        dtype=torch.long,
    )

    # Step 3: Single vectorized gather operation
    # Example: logprobs[row_indices, col_indices] -> [-2.1, -3.0, -2.2, -2.0, -1.4, -1.6]
    gathered_logprobs = logprobs[row_indices, col_indices]

    # Step 4: Split results back per request using torch operations
    # Example: split tensor [6] into chunks of sizes [2, 1, 3] -> [tensor(2), tensor(1), tensor(3)]
    split_logprobs = torch.split_with_sizes(
        gathered_logprobs, token_lengths.tolist(), dim=0
    )

    # Step 5: Format output to match expected return structure
    # Example: Convert split tensors back to list format with proper empty handling
    # i=0: [1,3] -> append split_logprobs[0] and [1,3]
    # i=1: [2] -> append split_logprobs[1] and [2]
    # i=2: [0,2,4] -> append split_logprobs[2] and [0,2,4]
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []

    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None and len(token_ids) > 0:
            output_token_ids_logprobs_val.append(split_logprobs[i])
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append(logprobs.new_empty(0))
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx


class OutputLogprobProcessor:
    """Output (decode) logprob processing: logprobs -> topk / token-ids /
    sampled-token gather, returned as a LogprobResult for the caller to
    write back onto LogitsProcessorOutput.

    Only logits/logprobs are needed here; sampler-side concerns (custom
    logit processors, NaN sanitizing) are injected via ``preprocess_fn``.
    """

    def compute_logprobs(
        self,
        logprobs: torch.Tensor,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        batch_next_token_ids: torch.Tensor,
    ) -> LogprobResult:
        # clamp to avoid -inf values
        logprobs.clamp_(min=torch.finfo(logprobs.dtype).min)

        result = LogprobResult()
        if any(x > 0 for x in top_logprobs_nums):
            (
                result.top_logprobs_val,
                result.top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums, no_copy_to_cpu=True)

        if any(x is not None for x in token_ids_logprobs):
            (
                result.token_ids_logprobs_val,
                result.token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(
                logprobs, token_ids_logprobs, no_copy_to_cpu=True
            )

        result.token_logprobs = logprobs[
            torch.arange(len(batch_next_token_ids), device=batch_next_token_ids.device),
            batch_next_token_ids,
        ]
        return result

    def compute_logprobs_only(
        self,
        next_token_logits: Optional[torch.Tensor],
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        preprocess_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Optional[LogprobResult]:
        """
        Compute logprobs for requested token IDs without performing sampling.

        Optimized for prefill-only scoring requests that need token probabilities
        but don't require next token generation.
        """

        if next_token_logits is None:
            logger.warning("No logits available for logprob computation")
            return None

        # Check if any requests actually need logprobs computation
        needs_token_ids_logprobs = any(
            token_ids is not None and len(token_ids) > 0
            for token_ids in token_ids_logprobs
        )
        needs_top_logprobs = any(x > 0 for x in top_logprobs_nums)

        if not (needs_token_ids_logprobs or needs_top_logprobs):
            return None

        # Preprocess logits (custom processors and NaN handling)
        logits = preprocess_fn(next_token_logits)

        # Compute logprobs
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        result = LogprobResult()
        # Handle top logprobs if requested
        if needs_top_logprobs:
            (
                result.top_logprobs_val,
                result.top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums, no_copy_to_cpu=True)

        # Handle token_ids logprobs if requested
        if needs_token_ids_logprobs:
            (
                result.token_ids_logprobs_val,
                result.token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_batch_optimized(logprobs, token_ids_logprobs)
        return result
