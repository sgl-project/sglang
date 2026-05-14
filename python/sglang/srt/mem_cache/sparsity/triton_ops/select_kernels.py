"""Double Sparsity selection — Q_label gather, score against K_label, top-k,
recency/sink union, dense fallback. Torch reference + Triton fast path.

Inputs (per layer, per decode step):
- queries          : [bs, num_q_heads_local, head_dim]    bf16/fp16
- channel_idx      : [num_kv_heads_local, S]              int32
- k_label_layer    : [num_tokens_in_pool, num_kv_heads, S] bf16/fp32
- req_to_token     : [max_reqs, max_ctx]                  int32 (logical → physical)
- req_pool_indices : [bs]                                 int64
- seq_lens         : [bs]                                 int64

Outputs:
- selected_logical : [bs, max_selected]  int32   logical token positions (padded with -1, ascending order)
- valid_lengths    : [bs]                int32   number of valid entries per row

Decode self-token invariant:
  K_label for the current decode position is written by attention_end *after*
  attention. Therefore stage-1 scoring covers history positions [0, seq_len-1)
  only; the current position (seq_len-1) is unconditionally retained via the
  recency window. `recent_tokens >= 1` is enforced at server startup.
"""

from __future__ import annotations

from typing import Tuple

import torch

GQA_REDUCTION_MAX_ABS = 0
GQA_REDUCTION_MEAN = 1
GQA_REDUCTION_SOQ = 2


def _compute_q_label(
    queries: torch.Tensor,
    channel_idx: torch.Tensor,
    *,
    num_kv_heads: int,
    gqa_reduction_id: int,
) -> torch.Tensor:
    """Gather S calibrated channels from Q and reduce across the GQA group.

    Returns `[bs, num_kv_heads, S]` matching `K_label`'s head axis. Expects
    `queries` already in `[bs, H_q, D]` form — the wrapper in
    `DoubleSparsityAlgorithm.retrieve_topk` reshapes from the 2D
    `[bs, H_q*D]` form attention layers commonly produce.
    """
    if queries.dim() != 3:
        raise ValueError(
            f"queries must be 3D [bs, H_q, D]; got shape {tuple(queries.shape)}. "
            f"Use DoubleSparsityAlgorithm.retrieve_topk to reshape from "
            f"the post-projection flat form."
        )
    bs, h_q, d = queries.shape
    if h_q % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads {h_q} not divisible by num_kv_heads {num_kv_heads}"
        )
    group = h_q // num_kv_heads
    s = channel_idx.shape[1]

    # Reshape Q so the GQA group axis is explicit: [bs, kv_heads, group, D]
    q = queries.view(bs, num_kv_heads, group, d)

    # Gather per-(kv_head, S) channels for each query in the group.
    # channel_idx: [kv_heads, S] → [1, kv_heads, 1, S]
    chans = channel_idx.to(torch.long).unsqueeze(0).unsqueeze(2)
    chans = chans.expand(bs, num_kv_heads, group, s)
    q_gathered = torch.gather(q, 3, chans)  # [bs, kv_heads, group, S]

    if gqa_reduction_id == GQA_REDUCTION_MAX_ABS:
        # Pick the channel-wise max-magnitude query, preserving sign.
        absvals = q_gathered.abs()
        max_idx = absvals.argmax(dim=2, keepdim=True)
        q_label = torch.gather(q_gathered, 2, max_idx).squeeze(2)
    elif gqa_reduction_id == GQA_REDUCTION_MEAN:
        q_label = q_gathered.mean(dim=2)
    elif gqa_reduction_id == GQA_REDUCTION_SOQ:
        # Sum-of-squares with sign of mean — energy biased reduction.
        sign = q_gathered.mean(dim=2).sign()
        energy = (q_gathered**2).sum(dim=2).sqrt()
        q_label = sign * energy
    else:
        raise ValueError(f"unknown gqa_reduction_id: {gqa_reduction_id}")
    return q_label


def _select_per_request_torch(
    *,
    q_label: torch.Tensor,  # [num_kv_heads, S]
    k_label_layer: torch.Tensor,  # [T, num_kv_heads, S]
    phys_tokens: torch.Tensor,  # [seq_len] int64 (req_to_token row)
    seq_len: int,
    token_budget: int,
    recent_tokens: int,
    sink_tokens: int,
    min_seq_len: int,
    max_selected: int,
) -> Tuple[torch.Tensor, int]:
    """Per-request selection. Returns (selected_logical[max_selected], valid_len)."""
    device = k_label_layer.device

    # Dense fallback: too short to be worth sparsifying.
    if seq_len < min_seq_len:
        out = torch.full((max_selected,), -1, dtype=torch.int32, device=device)
        n = min(seq_len, max_selected)
        out[:n] = torch.arange(n, dtype=torch.int32, device=device)
        return out, n

    # History-only scoring window: [0, seq_len - 1). Current position is
    # always retained via recency.
    history_len = max(seq_len - 1, 0)
    if history_len == 0:
        # Edge case: seq_len == 1; nothing to score, just keep position 0.
        out = torch.full((max_selected,), -1, dtype=torch.int32, device=device)
        out[0] = 0
        return out, 1

    # K_label[phys_tokens[:history_len]]: [history_len, kv_heads, S]
    phys_hist = phys_tokens[:history_len].to(torch.long)
    kl_rows = k_label_layer[phys_hist]

    # Score per (kv_head, position) then sum across heads → per-position score.
    # q_label: [kv_heads, S]; kl_rows: [history_len, kv_heads, S].
    # Per-head dot, then sum across heads → [history_len].
    scores_per_head = (kl_rows * q_label.unsqueeze(0)).sum(dim=-1)  # [hist, kv]
    scores = scores_per_head.sum(dim=1).float()  # reduce across kv_heads

    # Top-k over scored positions; budget capped to history_len.
    k = min(token_budget, history_len)
    topk_idx = scores.topk(k, sorted=False).indices.to(torch.int32)

    # Recency window: [seq_len - recent_tokens, seq_len), clamped to [0, seq_len).
    rec_start = max(seq_len - recent_tokens, 0)
    recent_idx = torch.arange(rec_start, seq_len, dtype=torch.int32, device=device)

    # Sink: [0, sink_tokens), clamped to [0, seq_len).
    sink_n = min(sink_tokens, seq_len)
    sink_idx = torch.arange(sink_n, dtype=torch.int32, device=device)

    # Union, dedup *logical*, sort ascending (logical order — never sorted by
    # physical id, per the plan to keep radix-prefix-reuse safe).
    combined = torch.cat([sink_idx, topk_idx, recent_idx])
    unique_sorted = torch.unique(combined, sorted=True)

    # Cap to max_selected: keep sink + recent (always), fill remaining slots
    # with the highest-scoring history positions, then re-sort logical-ascending.
    if unique_sorted.numel() > max_selected:
        is_boundary = (unique_sorted < sink_n) | (unique_sorted >= rec_start)
        forced = unique_sorted[is_boundary]
        history_only = unique_sorted[~is_boundary]
        if forced.numel() >= max_selected:
            unique_sorted = forced[:max_selected].sort().values
        else:
            slack = max_selected - forced.numel()
            hist_idx = history_only.to(torch.long).clamp_max(history_len - 1)
            hist_scores = scores[hist_idx]
            keep_top = hist_scores.topk(
                min(slack, hist_idx.numel()), sorted=False
            ).indices
            unique_sorted = torch.cat([forced, history_only[keep_top]]).sort().values

    n = unique_sorted.numel()
    out = torch.full((max_selected,), -1, dtype=torch.int32, device=device)
    out[:n] = unique_sorted.to(torch.int32)
    return out, n


def ds_select_tokens_torch_ref(
    *,
    queries: torch.Tensor,  # [bs, H_q_local, D]
    channel_idx: torch.Tensor,  # [H_kv_local, S]
    k_label_layer: torch.Tensor,  # [T, H_kv_local, S]
    req_to_token: torch.Tensor,  # [max_reqs, max_ctx]
    req_pool_indices: torch.Tensor,  # [bs]
    seq_lens: torch.Tensor,  # [bs]
    num_kv_heads: int,
    token_budget: int,
    recent_tokens: int,
    sink_tokens: int,
    min_seq_len: int,
    max_selected: int,
    gqa_reduction_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch CPU oracle for the full selection pipeline.

    **NOT a perf path.** Per-request Python loop with `seq_lens[b].item()`
    host syncs — used only by tests and CPU-only environments. The
    production CUDA path is `ds_select_tokens_triton` (two-stage Triton
    block-topk + score-aware union) in `select_triton.py`.
    """
    bs = queries.shape[0]
    device = queries.device

    q_label = _compute_q_label(
        queries,
        channel_idx,
        num_kv_heads=num_kv_heads,
        gqa_reduction_id=gqa_reduction_id,
    )  # [bs, kv_heads, S]

    out = torch.full((bs, max_selected), -1, dtype=torch.int32, device=device)
    valid = torch.zeros(bs, dtype=torch.int32, device=device)

    for b in range(bs):
        seq_len = int(seq_lens[b].item())
        if seq_len <= 0:
            continue
        phys = req_to_token[req_pool_indices[b].long(), :seq_len].to(torch.long)
        sel, n = _select_per_request_torch(
            q_label=q_label[b],
            k_label_layer=k_label_layer,
            phys_tokens=phys,
            seq_len=seq_len,
            token_budget=token_budget,
            recent_tokens=recent_tokens,
            sink_tokens=sink_tokens,
            min_seq_len=min_seq_len,
            max_selected=max_selected,
        )
        out[b] = sel
        valid[b] = n

    return out, valid


def ds_select_tokens_triton(
    *,
    queries: torch.Tensor,
    channel_idx: torch.Tensor,
    k_label_layer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    token_budget: int,
    recent_tokens: int,
    sink_tokens: int,
    min_seq_len: int,
    max_selected: int,
    gqa_reduction_id: int,
    block_t: int = 1024,
    k_block: int = 64,
    block_topk_logical: torch.Tensor = None,
    block_topk_scores: torch.Tensor = None,
    merged_logical: torch.Tensor = None,
    merged_scores: torch.Tensor = None,
    selected_logical: torch.Tensor = None,
    valid_lengths: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA perf path — two-stage block-topk + score-aware union.

    Pipeline (matches the v1.1 plan):
      1. Stage 1: per (bs, kv_head, block) Triton program scores a
         BLOCK_T-token tile of K_label and emits local top-K_BLOCK
         (logical, score) pairs. Bounded by per-row seq_lens.
      2. Stage 2: per (bs, kv_head) Triton program merges
         num_blocks * K_BLOCK candidates down to effective_budget by
         score.
      3. Union: per-batch torch-on-CUDA pass deduplicates across
         kv_heads (max-score merge), drops always-keep overlaps,
         caps to max_selected_per_request by SCORE (never logical
         position), appends sink + recency, sorts logical-ascending.

    All capture-safe: stage-1/2 are static-grid Triton; union is
    bounded-shape torch with preallocated outputs (capacity-guarded
    by `DoubleSparsityRuntimeConfig.warn_capacity` at startup).

    Optional buffer kwargs (`block_topk_logical`, `block_topk_scores`,
    `merged_logical`, `merged_scores`, `selected_logical`,
    `valid_lengths`) let the caller pre-allocate once at init and
    reuse across decode steps. The helpers write in-place when these
    are provided, so the production path performs zero output-tensor
    allocation. When `None`, each helper falls back to allocating its
    own tensor (used by tests and one-shot calls).
    """
    if not queries.is_cuda:
        raise RuntimeError("ds_select_tokens_triton requires CUDA tensors")

    from sglang.srt.mem_cache.sparsity.triton_ops.select_triton import (
        ds_select_stage1_block_topk,
        ds_select_stage2_merge,
        ds_union_per_batch,
    )

    max_ctx = req_to_token.shape[1]
    num_blocks = (max_ctx + block_t - 1) // block_t
    effective_budget = min(token_budget, num_blocks * k_block)

    # Stage 1
    block_topk_logical, block_topk_scores = ds_select_stage1_block_topk(
        queries=queries,
        channel_idx=channel_idx,
        k_label=k_label_layer,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        num_kv_heads=num_kv_heads,
        block_t=block_t,
        k_block=k_block,
        gqa_reduction_id=gqa_reduction_id,
        block_topk_logical=block_topk_logical,
        block_topk_scores=block_topk_scores,
    )

    # Stage 2
    merged_logical, merged_scores = ds_select_stage2_merge(
        block_topk_logical=block_topk_logical,
        block_topk_scores=block_topk_scores,
        effective_budget=effective_budget,
        merged_logical=merged_logical,
        merged_scores=merged_scores,
    )

    # Union
    return ds_union_per_batch(
        merged_logical=merged_logical,
        merged_scores=merged_scores,
        seq_lens=seq_lens,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
        min_seq_len=min_seq_len,
        max_selected_per_request=max_selected,
        selected_logical=selected_logical,
        valid_lengths=valid_lengths,
    )
