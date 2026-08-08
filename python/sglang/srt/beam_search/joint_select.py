"""Joint token selection for beam search: pure tensor functions.

Contract (overlap / cuda-graph ready): tensor-in / tensor-out, fixed output
shapes for a given (num_rows, num_candidates, beam_width) signature, no D2H
sync, no data-dependent host branches (the only python branch keys on a
static tensor shape). The caller owns the single sync point per step.

Selection semantics (ported from the original expansion loop): walk the top
num_candidates extensions in descending cumulative-logprob order; stop-token
candidates finish, non-stop candidates survive; a candidate is examined only
while fewer than beam_width survivors precede it in score order.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SelectResult:
    """Fixed-shape outputs of one expansion step.

    Survivor slots are valid in [0, num_survivors); finished slots in
    [0, num_finished); the remaining slots hold zeros from the dump-slot
    scatter and must not be read.
    """

    next_tokens: torch.Tensor  # [beam_width] int64
    parent_idx: torch.Tensor  # [beam_width] int64, row index into the input frontier
    new_cum_logprobs: torch.Tensor  # [beam_width] float32
    num_survivors: torch.Tensor  # [] int64; < beam_width means the group must finish
    fin_tokens: torch.Tensor  # [num_candidates] int64
    fin_parent_idx: torch.Tensor  # [num_candidates] int64
    fin_cum_logprobs: torch.Tensor  # [num_candidates] float32
    num_finished: torch.Tensor  # [] int64


@dataclass
class FinalSelect:
    """Top-beam_width candidates of a length-terminated step (all finished)."""

    tokens: torch.Tensor  # [beam_width] int64
    parent_idx: torch.Tensor  # [beam_width] int64
    cum_logprobs: torch.Tensor  # [beam_width] float32


def _scatter_fixed(src: torch.Tensor, slot: torch.Tensor, size: int) -> torch.Tensor:
    # Fixed-shape compaction: element i lands at slot[i]; non-selected elements
    # all target the dump slot `size`, which is sliced away.
    buf = src.new_zeros(size + 1)
    buf.scatter_(0, slot, src)
    return buf[:size]


def _ranked_candidates(cum_logprobs, top_logprobs, top_tokens, num_out):
    """Score all row x candidate extensions, return the top num_out sorted."""
    num_candidates = top_logprobs.shape[1]
    scores = cum_logprobs.unsqueeze(1) + top_logprobs
    cand_scores, cand_idx = scores.reshape(-1).topk(num_out, sorted=True)
    parent = cand_idx // num_candidates
    tokens = top_tokens.reshape(-1).gather(0, cand_idx)
    return cand_scores, parent, tokens


def joint_select(
    cum_logprobs: torch.Tensor,  # [num_rows] float32, frontier cumulative logprobs
    top_logprobs: torch.Tensor,  # [num_rows, num_candidates] float32
    top_tokens: torch.Tensor,  # [num_rows, num_candidates] int64
    stop_token_ids: torch.Tensor,  # [num_stop] int64, may be empty (ignore_eos)
    beam_width: int,
) -> SelectResult:
    num_candidates = top_logprobs.shape[1]
    k = beam_width

    cand_scores, parent, tokens = _ranked_candidates(
        cum_logprobs, top_logprobs, top_tokens, num_candidates
    )

    if stop_token_ids.numel() > 0:  # static-shape branch, constant under capture
        is_stop = torch.isin(tokens, stop_token_ids)
    else:
        is_stop = torch.zeros_like(tokens, dtype=torch.bool)

    non_stop = ~is_stop
    non_stop_rank = non_stop.long().cumsum(0)  # 1-based at non-stop positions
    survivor = non_stop & (non_stop_rank <= k)
    # Examined while fewer than k survivors strictly precede the candidate.
    examined = (non_stop_rank - non_stop.long()) < k
    finished = is_stop & examined
    fin_rank = finished.long().cumsum(0)

    surv_slot = torch.where(
        survivor, non_stop_rank - 1, torch.full_like(non_stop_rank, k)
    )
    fin_slot = torch.where(
        finished, fin_rank - 1, torch.full_like(fin_rank, num_candidates)
    )

    return SelectResult(
        next_tokens=_scatter_fixed(tokens, surv_slot, k),
        parent_idx=_scatter_fixed(parent, surv_slot, k),
        new_cum_logprobs=_scatter_fixed(cand_scores, surv_slot, k),
        num_survivors=survivor.long().sum(),
        fin_tokens=_scatter_fixed(tokens, fin_slot, num_candidates),
        fin_parent_idx=_scatter_fixed(parent, fin_slot, num_candidates),
        fin_cum_logprobs=_scatter_fixed(cand_scores, fin_slot, num_candidates),
        num_finished=finished.long().sum(),
    )


def select_final_topk(
    cum_logprobs: torch.Tensor,  # [num_rows] float32
    top_logprobs: torch.Tensor,  # [num_rows, num_candidates] float32
    top_tokens: torch.Tensor,  # [num_rows, num_candidates] int64
    beam_width: int,
) -> FinalSelect:
    """Length-terminated step: the best beam_width extensions all finish.

    No stop check is needed; the caller decides this step deterministically
    from the step counter (max_new_tokens is uniform across the group).
    """
    cand_scores, parent, tokens = _ranked_candidates(
        cum_logprobs, top_logprobs, top_tokens, beam_width
    )
    return FinalSelect(tokens=tokens, parent_idx=parent, cum_logprobs=cand_scores)
