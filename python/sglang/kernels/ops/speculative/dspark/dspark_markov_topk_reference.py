# SPDX-License-Identifier: Apache-2.0
"""Readable, independent oracle for the DSpark candidate proposal distribution.

This module deliberately imports only Torch. It is usable on CPU without the
SGLang runtime, Triton, FlashInfer, a checkpoint, or a CUDA device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MarkovCandidateResult:
    tokens: torch.Tensor
    corrected_logits: torch.Tensor
    prev_tokens: torch.Tensor


def validate_mapping(
    d2t_offset: Optional[torch.Tensor], draft_vocab_size: int, target_vocab_size: int
) -> None:
    """Validate once at initialization, including injectivity, never at replay."""
    if target_vocab_size <= 0 or draft_vocab_size <= 0:
        raise ValueError("DSpark vocabulary sizes must be positive")
    if d2t_offset is None:
        if draft_vocab_size != target_vocab_size:
            raise ValueError("Reduced DSpark vocabulary requires a d2t offset mapping")
        return
    if d2t_offset.shape != (draft_vocab_size,) or d2t_offset.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("DSpark d2t must be an integer offset vector of length Vd")
    mapped = torch.arange(draft_vocab_size, device=d2t_offset.device) + d2t_offset
    if bool(((mapped < 0) | (mapped >= target_vocab_size)).any()):
        raise ValueError("DSpark d2t mapping contains out-of-range target token IDs")
    if mapped.unique().numel() != draft_vocab_size:
        raise ValueError("DSpark d2t mapping must be injective (one-to-one)")


def candidate_uniform(seed: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """A token-keyed random stream shared by the reference and fused kernel.

    Seeds are fresh independent Torch RNG draws for each request and draft step.
    The integer permutation produces one 23-bit open-interval uniform per target
    token. Candidate ordering therefore never changes a token's noise. This is
    a draft-only stream; verifier RNG draws must be independent.
    """
    mask = 0xFFFFFFFF
    x = (target_ids.to(torch.int64) ^ seed.to(torch.int64) ^ 0xA511E9B3) & mask
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & mask
    x = ((x ^ (x >> 15)) * 0x846CA68B) & mask
    x = x ^ (x >> 16)
    return ((x >> 9).float() + 0.5) * (1.0 / 8388608.0)


@torch.no_grad()
def markov_candidates_reference(
    base_logits: torch.Tensor,
    anchor: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    alpha: float,
    topk: int,
    bias_topk: int,
    target_vocab_size: int,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    d2t_offset: Optional[torch.Tensor] = None,
    seeds: Optional[torch.Tensor] = None,
) -> MarkovCandidateResult:
    """Derive candidate sets, canonical scores, and the whole chain from inputs.

    The static side is authoritative when the two sets overlap. This oracle
    recomputes its static projections from raw weights, rather than accepting
    realized scores or the optimized kernel's static table as ground truth.
    Tied winners are resolved by the smallest *target* token ID.
    """
    if base_logits.ndim != 3:
        raise ValueError("base_logits must have shape [B, N, Vd]")
    bs, steps, vd = base_logits.shape
    if not (0 < topk <= vd and 0 <= bias_topk <= vd):
        raise ValueError("Candidate reference requires 0 < K <= Vd and 0 <= M <= Vd")
    if w1.ndim != 2 or w2.shape != (vd, w1.shape[1]):
        raise ValueError("DSpark Markov weight shapes do not match Vd/rank")
    if w1.shape[0] < target_vocab_size:
        raise ValueError("DSpark W1 must cover every predecessor target token")
    validate_mapping(d2t_offset, vd, target_vocab_size)
    if anchor.shape != (bs,):
        raise ValueError("anchor must have shape [B]")
    if bool(((anchor < 0) | (anchor >= target_vocab_size)).any()):
        raise ValueError("DSpark anchor contains an invalid target token ID")
    if temperatures.numel() != bs or greedy_mask.numel() != bs:
        raise ValueError("temperatures and greedy_mask must have one value per row")
    temperatures = temperatures.reshape(bs)
    greedy_mask = greedy_mask.reshape(bs)
    if bool(((temperatures <= 0) & ~greedy_mask.bool()).any()):
        raise ValueError("Probabilistic DSpark rows require positive temperatures")
    if seeds is None:
        seeds = torch.randint(0, 2**31, (bs, steps), device=base_logits.device)
    elif seeds.shape != (bs, steps):
        raise ValueError("DSpark proposal seeds must have shape [B, N]")
    cache = torch.full(
        (bs, steps, target_vocab_size),
        float("-inf"),
        dtype=torch.float32,
        device=base_logits.device,
    )
    tokens = torch.empty((bs, steps), dtype=torch.int64, device=base_logits.device)
    predecessors = torch.empty_like(tokens)
    wf1, wf2 = w1.float(), w2.float()
    for row in range(bs):
        prev = int(anchor[row])
        for step in range(steps):
            predecessors[row, step] = prev
            base = base_logits[row, step].float()
            base_ids = torch.topk(base, topk).indices
            # Full projection here makes the oracle independent of online
            # candidate gathering and cached values used by the implementation.
            dense_bias = (wf1[prev : prev + 1] @ wf2.T).flatten()
            if bias_topk:
                if alpha == 0:
                    static_ids = torch.arange(bias_topk, device=base.device)
                else:
                    static_ids = torch.topk(
                        dense_bias, bias_topk, largest=alpha > 0
                    ).indices
                base_ids = base_ids[~torch.isin(base_ids, static_ids)]
                ids = torch.cat((base_ids, static_ids))
            else:
                ids = base_ids
            # Independent FP32 projection; exact small integer test fixtures
            # avoid conflating accumulation-order differences with semantic bugs.
            scores = base[ids] + float(alpha) * dense_bias[ids]
            target_ids = ids if d2t_offset is None else ids + d2t_offset[ids]
            if not bool(torch.isfinite(scores).any()):
                # Match the fused kernel's well-defined degenerate-row proposal.
                winner = target_ids.min()
                cache[row, step, winner] = 0.0
            else:
                cache[row, step, target_ids] = scores
                if bool(greedy_mask[row]):
                    keys = scores
                else:
                    uniform = candidate_uniform(seeds[row, step], target_ids)
                    keys = scores / temperatures[row] - torch.log(-torch.log(uniform))
                winner = target_ids[keys == keys.max()].min()
            tokens[row, step] = winner
            prev = int(winner)
    return MarkovCandidateResult(tokens, cache, predecessors)
