"""Greedy Markov proposals restricted to base and bias candidate IDs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _markov_candidate_step_kernel(
    base_ptr,
    base_ids_ptr,
    prev_ptr,
    embed_ptr,
    w2_ptr,
    table_ptr,
    output_ptr,
    base_row_stride,
    ids_row_stride,
    prev_stride,
    embed_row_stride,
    w2_row_stride,
    table_row_stride,
    V: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK_C)
    prev = tl.load(prev_ptr + row * prev_stride).to(tl.int64)
    prev_ok = (prev >= 0) & (prev < V)
    base_id = tl.load(
        base_ids_ptr + row * ids_row_stride + lane, mask=lane < K, other=0
    ).to(tl.int64)
    bias_id = tl.load(
        table_ptr + prev * table_row_stride + lane - K,
        mask=prev_ok & (lane >= K) & (lane < K + M),
        other=0,
    ).to(tl.int64)
    candidate = tl.where(lane < K, base_id, bias_id)
    valid = (lane < K + M) & (candidate >= 0) & (candidate < V)
    valid &= (lane < K) | prev_ok
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for r0 in range(0, R, BLOCK_R):
        rank = r0 + tl.arange(0, BLOCK_R)
        embed = tl.load(
            embed_ptr + row * embed_row_stride + rank, mask=rank < R, other=0.0
        ).to(tl.float32)
        weight = tl.load(
            w2_ptr + candidate[:, None] * w2_row_stride + rank[None, :],
            mask=valid[:, None] & (rank[None, :] < R),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(weight * embed[None, :], axis=1)
    base = tl.load(
        base_ptr + row * base_row_stride + candidate,
        mask=valid,
        other=float("-inf"),
    ).to(tl.float32)
    scores = tl.where(valid, base + acc, float("-inf"))
    nan = valid & (scores != scores)
    has_nan = tl.sum(nan.to(tl.int32), axis=0) > 0
    best = tl.max(tl.where(nan, float("-inf"), scores), axis=0)
    wins = valid & tl.where(has_nan, nan, scores == best)
    selected = tl.min(tl.where(wins, candidate, V), axis=0)
    tl.store(output_ptr + row, selected)


def markov_candidate_step(
    base_logits: torch.Tensor,
    base_ids: torch.Tensor,
    prev_tokens: torch.Tensor,
    prev_embeds: torch.Tensor,
    w2_weight: torch.Tensor,
    bias_top_ids: torch.Tensor,
) -> torch.Tensor:
    """Select the lowest-ID maximum over a restricted Markov candidate union.

    Args:
        base_logits: Dense CUDA floating-point logits, shape [B, V].
        base_ids: Integer base candidate IDs, shape [B, K].
        prev_tokens: Integer previous token IDs, shape [B]; may be strided.
        prev_embeds: Floating-point previous-token embeddings, shape [B, R].
        w2_weight: Dense floating-point projection weights, shape [V, R].
        bias_top_ids: Integer bias candidate table, shape [V, M].

    Returns:
        CUDA int64 token IDs of shape [B]. Scores accumulate in FP32 in
        32-rank chunks. Exact ties select the lowest token ID; NaN scores
        select the lowest candidate ID with a NaN score.

    Candidate and previous-token IDs must be in [0, V). Value checks belong
    to the caller so this function remains graph-safe. Duplicate candidates
    are allowed. This computes a subset argmax, not a full-vocabulary argmax.
    """
    tensors = (
        base_logits,
        base_ids,
        prev_tokens,
        prev_embeds,
        w2_weight,
        bias_top_ids,
    )
    if any(t.layout != torch.strided or not t.is_cuda for t in tensors):
        raise ValueError("Markov candidates require dense CUDA tensors")
    if any(t.device != base_logits.device for t in tensors):
        raise ValueError("Markov candidate tensors must share a CUDA device")
    if (
        base_logits.ndim != 2
        or base_ids.ndim != 2
        or prev_tokens.ndim != 1
        or prev_embeds.ndim != 2
        or w2_weight.ndim != 2
        or bias_top_ids.ndim != 2
    ):
        raise ValueError("Expected 2D inputs and 1D previous tokens")
    batch, vocab = base_logits.shape
    rank = w2_weight.shape[1]
    k, m = base_ids.shape[1], bias_top_ids.shape[1]
    if vocab <= 0 or rank <= 0 or not (1 <= k <= 128 and 1 <= m <= 128):
        raise ValueError("Require positive vocab/rank and K/M in [1, 128]")
    if (
        base_ids.shape[0] != batch
        or prev_tokens.shape != (batch,)
        or prev_embeds.shape != (batch, rank)
        or w2_weight.shape[0] != vocab
        or bias_top_ids.shape[0] != vocab
    ):
        raise ValueError("Inconsistent batch, vocabulary, or rank dimensions")
    float_types = (torch.bfloat16, torch.float16, torch.float32)
    if any(t.dtype not in float_types for t in (base_logits, prev_embeds, w2_weight)):
        raise ValueError(
            "Markov logits, embeddings, and weights must be BF16/FP16/FP32"
        )
    if any(
        t.dtype not in (torch.int32, torch.int64)
        for t in (base_ids, prev_tokens, bias_top_ids)
    ):
        raise ValueError("Markov candidate and previous-token IDs must be int32/int64")
    if any(
        t.stride(1) != 1
        for t in (base_logits, base_ids, prev_embeds, w2_weight, bias_top_ids)
    ):
        raise ValueError("Markov candidate 2D inputs must have unit innermost stride")
    output = torch.empty((batch,), dtype=torch.int64, device=base_logits.device)
    if batch:
        _markov_candidate_step_kernel[(batch,)](
            base_logits,
            base_ids,
            prev_tokens,
            prev_embeds,
            w2_weight,
            bias_top_ids,
            output,
            base_logits.stride(0),
            base_ids.stride(0),
            prev_tokens.stride(0),
            prev_embeds.stride(0),
            w2_weight.stride(0),
            bias_top_ids.stride(0),
            V=vocab,
            R=rank,
            K=k,
            M=m,
            BLOCK_C=triton.next_power_of_2(k + m),
            BLOCK_R=32,
        )
    return output
