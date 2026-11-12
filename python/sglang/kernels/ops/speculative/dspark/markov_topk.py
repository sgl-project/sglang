# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Greedy candidate Markov walk, fused across draft positions.

Adapted from vLLM's DSpark ``topk_markov.py`` for FP32 scores and target-ID ties.
"""

import math

import torch
import triton
import triton.language as tl


def can_fuse_markov_walk(w1, w2, base_logits, topk: int) -> bool:
    """Whether the NVIDIA kernel supports these weight and logit layouts."""
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    return (
        torch.version.hip is None
        and base_logits.is_cuda
        and w1.device == w2.device == base_logits.device
        and w1.dtype in dtypes
        and w2.dtype in dtypes
        and base_logits.dtype in dtypes
        and w1.ndim == w2.ndim == 2
        and base_logits.ndim == 3
        and w1.stride(1) == w2.stride(1) == 1
        and 0 < w1.shape[1] == w2.shape[1] <= 1024
        and w1.shape[0] > 0
        and w2.shape[0] >= base_logits.shape[-1]
        and 0 < topk <= min(128, base_logits.shape[-1])
    )


@triton.jit
def _markov_walk_kernel(
    LOGITS,
    BASE_VALUES,
    BASE_IDS,
    MARKOV_IDS,
    MARKOV_BIASES,
    W1,
    W2,
    ANCHORS,
    D2T,
    OUT,
    LOGITS_BS: tl.constexpr,
    LOGITS_GS: tl.constexpr,
    LOGITS_VS: tl.constexpr,
    W1S: tl.constexpr,
    W2S: tl.constexpr,
    ANCHOR_S: tl.constexpr,
    D2T_S: tl.constexpr,
    SCALE: tl.constexpr,
    STEPS: tl.constexpr,
    K: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_D2T: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    valid = cols < K
    prev = tl.load(ANCHORS + req * ANCHOR_S).to(tl.int64)
    no_id = 0x7FFFFFFFFFFFFFFF

    for step in range(STEPS):
        base_offset = (req * STEPS + step) * K + cols
        base_ids = tl.load(BASE_IDS + base_offset, valid, other=0).to(tl.int64)
        bias_ids = tl.load(MARKOV_IDS + prev * K + cols, valid, other=0).to(tl.int64)
        # The cached bias owns overlapping IDs, just as the second scatter does.
        overlap = (
            tl.sum(
                ((base_ids[:, None] == bias_ids[None, :]) & valid[None, :]).to(
                    tl.int32
                ),
                axis=1,
            )
            > 0
        )
        base_valid = valid & ~overlap
        base_scores = tl.load(BASE_VALUES + base_offset, valid, other=0).to(tl.float32)
        if SCALE != 0.0:
            correction = tl.full((BLOCK_K,), 0, tl.float32)
            for start in range(0, RANK, BLOCK_R):
                ranks = start + tl.arange(0, BLOCK_R)
                embed = tl.load(W1 + prev * W1S + ranks, ranks < RANK, 0).to(tl.float32)
                weight = tl.load(
                    W2 + base_ids[:, None] * W2S + ranks[None, :],
                    base_valid[:, None] & (ranks[None, :] < RANK),
                    other=0,
                ).to(tl.float32)
                correction += tl.sum(weight * embed[None, :], axis=1)
            base_scores += correction * SCALE
        bias_base = tl.load(
            LOGITS + req * LOGITS_BS + step * LOGITS_GS + bias_ids * LOGITS_VS,
            valid,
            other=0,
        ).to(tl.float32)
        bias = tl.load(MARKOV_BIASES + prev * K + cols, valid, other=0)
        scores = tl.cat(
            tl.where(base_valid, base_scores, -float("inf")),
            tl.where(valid, bias_base + bias, -float("inf")),
        )
        ids = tl.cat(base_ids, bias_ids)
        selected = tl.cat(base_valid, valid)
        if HAS_D2T:
            ids += tl.load(D2T + ids * D2T_S, selected, other=0)
        target_ids = tl.where(selected, ids, no_id)
        finite = (
            tl.sum(
                ((scores > -float("inf")) & (scores < float("inf"))).to(tl.int32),
                axis=0,
            )
            > 0
        )
        # With finite scores, NaNs win argmax; ties use the lowest target ID.
        nan_id = tl.min(tl.where(scores != scores, target_ids, no_id), axis=0)
        maximum = tl.max(tl.where(scores != scores, -float("inf"), scores), axis=0)
        winner = tl.min(tl.where(scores == maximum, target_ids, no_id), axis=0)
        winner = tl.where(nan_id != no_id, nan_id, winner)
        winner = tl.where(finite, winner, tl.min(target_ids, axis=0))
        tl.store(OUT + req * STEPS + step, winner)
        prev = winner


def markov_walk_topk(
    *,
    base_logits: torch.Tensor,
    base_values: torch.Tensor,
    base_ids: torch.Tensor,
    markov_ids: torch.Tensor,
    markov_biases: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    anchor_tokens: torch.Tensor,
    d2t: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Return greedy target IDs [batch, steps] with one program per request.

    Base logits are [batch, steps, draft_vocab]; top-k values/IDs are contiguous
    [batch, steps, k]. Markov IDs and scaled FP32 biases are contiguous
    [predecessor_vocab, k]. The caller validates token IDs and the injective d2t
    offset mapping before capture. FP32 reductions can differ from torch near
    ties; this kernel does not promise bitwise-equivalent matrix multiplication.
    """
    if base_ids.ndim != 3:
        raise ValueError("Markov base IDs must have shape [batch, steps, k].")
    batch, steps, k = base_ids.shape
    if not can_fuse_markov_walk(w1, w2, base_logits, k) or not math.isfinite(scale):
        raise ValueError("Unsupported fused Markov weight/logit layout or scale.")
    index_dtypes = (torch.int32, torch.int64)
    if not (
        base_logits.shape[:2] == (batch, steps)
        and base_values.shape == base_ids.shape
        and base_values.dtype == base_logits.dtype
        and base_ids.dtype in index_dtypes
        and markov_ids.shape == (w1.shape[0], k)
        and markov_ids.dtype in index_dtypes
        and markov_biases.shape == markov_ids.shape
        and markov_biases.dtype == torch.float32
        and anchor_tokens.shape == (batch,)
        and anchor_tokens.dtype in index_dtypes
    ):
        raise ValueError("Incompatible fused Markov candidate metadata.")
    tensors = (base_values, base_ids, markov_ids, markov_biases)
    if not all(t.is_contiguous() for t in tensors) or not all(
        t.device == base_logits.device for t in (*tensors, anchor_tokens)
    ):
        raise ValueError(
            "Markov candidate tables must be contiguous on the logit device."
        )
    if d2t is not None and not (
        d2t.shape == (base_logits.shape[-1],)
        and d2t.dtype in index_dtypes
        and d2t.device == base_logits.device
    ):
        raise ValueError("Markov d2t must be a draft-vocabulary offset vector.")
    out = torch.empty((batch, steps), dtype=torch.int64, device=base_logits.device)
    if batch == 0 or steps == 0:
        return out
    rank = w1.shape[1]
    _markov_walk_kernel[(batch,)](
        base_logits,
        base_values,
        base_ids,
        markov_ids,
        markov_biases,
        w1,
        w2,
        anchor_tokens,
        d2t,
        out,
        *base_logits.stride(),
        w1.stride(0),
        w2.stride(0),
        anchor_tokens.stride(0),
        d2t.stride(0) if d2t is not None else 0,
        scale,
        steps,
        k,
        rank,
        triton.next_power_of_2(k),
        min(triton.next_power_of_2(rank), 64),
        d2t is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
