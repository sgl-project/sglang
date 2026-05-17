"""Double Sparsity selection — two-stage block-topk Triton perf path.

This module hosts the v1.1 perf-path selection kernels:

  * Stage 1: `ds_select_stage1_block_topk` — per (bs, kv_head, block)
    Triton program scores a `BLOCK_T`-token tile of K_label against the
    inline-gathered Q_label and emits the local top-`K_BLOCK` (logical
    position, score) pairs.

  * Stage 2: `ds_select_stage2_merge` — (next commit) per (bs, kv_head)
    program merges across blocks down to `effective_budget` candidates.

  * Union: `ds_union_per_batch` — (next commit) score-aware Triton
    kernel; deduplicates across kv_heads, applies sink/recency, caps to
    `max_selected_per_request` by score.

The single-program "fused" kernel from the v1 design is debug-only and
does NOT live in the perf path; the placeholder
`ds_select_tokens_torch_ref_vectorized` over `req_to_token.shape[1]`
(max_ctx) was the source of the v1 misleading bench numbers and is
removed in v1.1-7.

Approximate top-k by construction. With `K_BLOCK ≤ token_budget`, a
token whose score is globally top-N but ranked `> K_BLOCK` within its
block is dropped. Tests use a block-topk oracle (NOT `torch.topk`) for
parity, plus a separate recall regression detector.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

# GQA reduction codes (matched in select_kernels.py for cross-imports).
from sglang.srt.mem_cache.sparsity.triton_ops.select_kernels import (
    GQA_REDUCTION_MAX_ABS,
    GQA_REDUCTION_MEAN,
    GQA_REDUCTION_SOQ,
)

# --------------------------------------------------------------------------- #
# Torch reference — block-topk oracle, NOT a perf path.                       #
# --------------------------------------------------------------------------- #


def _q_label_for_kv_head(
    queries: torch.Tensor,  # [bs, H_q, D]
    channel_idx: torch.Tensor,  # [H_kv, S] int32
    *,
    bs_idx: int,
    kv_idx: int,
    num_kv_heads: int,
    gqa_reduction_id: int,
) -> torch.Tensor:
    """Reference Q_label gather + GQA reduction for a single (bs, kv_head).

    Equivalent to what the Triton kernel does inline. Used by the parity
    oracle — the kernel takes the raw queries tensor and computes Q_label
    per program, never materializes a `[bs, H_kv, S]` tensor.
    """
    bs, h_q, d = queries.shape
    group = h_q // num_kv_heads
    s = channel_idx.shape[1]
    # [group, S]
    chans = channel_idx[kv_idx].long()
    q_group = queries[bs_idx, kv_idx * group : (kv_idx + 1) * group, :]  # [group, D]
    q_gathered = q_group[:, chans]  # [group, S]

    if gqa_reduction_id == GQA_REDUCTION_MAX_ABS:
        # Per channel, pick the value with max |abs| across the group.
        absvals = q_gathered.abs()
        max_idx = absvals.argmax(dim=0, keepdim=True)
        return torch.gather(q_gathered, 0, max_idx).squeeze(0)
    if gqa_reduction_id == GQA_REDUCTION_MEAN:
        return q_gathered.mean(dim=0)
    if gqa_reduction_id == GQA_REDUCTION_SOQ:
        sign = q_gathered.mean(dim=0).sign()
        energy = (q_gathered**2).sum(dim=0).sqrt()
        return sign * energy
    raise ValueError(f"unknown gqa_reduction_id: {gqa_reduction_id}")


def ds_select_stage1_block_topk_torch_ref(
    *,
    queries: torch.Tensor,  # [bs, H_q, D]
    channel_idx: torch.Tensor,  # [H_kv, S] int32
    k_label: torch.Tensor,  # [num_tokens_in_pool, H_kv, S]
    req_to_token: torch.Tensor,  # [max_reqs, max_ctx]
    req_pool_indices: torch.Tensor,  # [bs] int64
    seq_lens: torch.Tensor,  # [bs] int64
    num_kv_heads: int,
    block_t: int,
    k_block: int,
    gqa_reduction_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference impl of the stage-1 block-topk algorithm.

    Returns:
        block_topk_logical: `[bs, H_kv, num_blocks, k_block]` int32.
            Logical token positions per (bs, kv_head, block); -1 for
            slots in blocks past history.
        block_topk_scores: same shape, fp32. NEG_INF for invalid slots.

    Algorithm — must match the Triton kernel byte-for-byte:
      For each (bs, kv_head, block):
        1. Inline Q_label gather + GQA reduction → [S].
        2. For tokens t in [block_start, block_start + BLOCK_T):
           - if t >= seq_len - 1: score = -inf (history-only).
           - phys = req_to_token[req_pool_indices[bs], t].
           - score = sum_s Q_label[s] * K_label[phys, kv_head, s].
        3. Top-k_block of (score, position) by score, descending.
           Ties broken by lowest position (stable).
    """
    bs, h_q, _ = queries.shape
    h_kv = num_kv_heads
    max_ctx = req_to_token.shape[1]
    num_blocks = (max_ctx + block_t - 1) // block_t
    device = queries.device

    out_logical = torch.full(
        (bs, h_kv, num_blocks, k_block), -1, dtype=torch.int32, device=device
    )
    out_scores = torch.full(
        (bs, h_kv, num_blocks, k_block),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )

    for b in range(bs):
        seq_len = int(seq_lens[b].item())
        history_len = max(seq_len - 1, 0)
        rpi = int(req_pool_indices[b].item())
        for h in range(h_kv):
            q_label = _q_label_for_kv_head(
                queries,
                channel_idx,
                bs_idx=b,
                kv_idx=h,
                num_kv_heads=num_kv_heads,
                gqa_reduction_id=gqa_reduction_id,
            ).to(
                torch.float32
            )  # [S]

            for blk in range(num_blocks):
                block_start = blk * block_t
                if block_start >= history_len:
                    continue  # entire block past history; leave -1 / -inf
                t_lo, t_hi = block_start, min(block_start + block_t, history_len)
                logicals = torch.arange(t_lo, t_hi, device=device)
                phys = req_to_token[rpi, logicals].long()
                kl = k_label[phys, h, :].to(torch.float32)  # [t_count, S]
                scores = (kl * q_label[None, :]).sum(dim=1)  # [t_count]

                # Top-k_block by score, stable-by-position.
                # torch.topk is unstable; emulate stable by sort + take first
                # k_block on the "score descending, position ascending" key.
                k = min(k_block, scores.numel())
                # Sort by score descending; ties: position ascending. Triton
                # kernel implements this via iterative argmax + min-position
                # tie-break, which is equivalent.
                order = torch.argsort(scores, descending=True, stable=True)[:k]
                top_logical = logicals[order]
                top_scores = scores[order]
                out_logical[b, h, blk, :k] = top_logical.to(torch.int32)
                out_scores[b, h, blk, :k] = top_scores

    return out_logical, out_scores


# --------------------------------------------------------------------------- #
# Triton kernel — stage-1 block-topk, perf path.                              #
# --------------------------------------------------------------------------- #


@triton.jit
def _ds_select_stage1_block_topk_kernel(
    Q_ptr,
    channel_idx_ptr,
    k_label_ptr,
    req_to_token_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    block_topk_logical_ptr,
    block_topk_scores_ptr,
    # Q strides ([bs, H_q, D]; assumed last-dim contiguous so D-stride is 1)
    Q_stride_b,
    Q_stride_h,
    # channel_idx strides ([H_kv, S]; last-dim contiguous)
    chan_stride_h,
    # k_label strides ([T, H_kv, S]; last-dim contiguous)
    kl_stride_t,
    kl_stride_h,
    # req_to_token strides ([max_reqs, max_ctx])
    r2t_stride_r,
    # output strides ([bs, H_kv, num_blocks, K_BLOCK]; last-dim contiguous)
    btl_stride_b,
    btl_stride_h,
    btl_stride_blk,
    bts_stride_b,
    bts_stride_h,
    bts_stride_blk,
    # constexprs
    H_q: tl.constexpr,
    H_kv: tl.constexpr,
    GROUP: tl.constexpr,  # H_q // H_kv
    D: tl.constexpr,
    S: tl.constexpr,
    BLOCK_T: tl.constexpr,
    K_BLOCK: tl.constexpr,
    GQA_REDUCTION: tl.constexpr,
):
    """One program per (bs_idx, kv_idx, blk_idx).

    Streams a BLOCK_T-token tile of K_label, scores against the inline-
    gathered Q_label, picks top-K_BLOCK by iterative argmax. Tokens past
    `seq_lens[bs_idx] - 1` are masked to NEG_INF (history-only).
    """
    NEG_INF = -1e38

    bs_idx = tl.program_id(0)
    kv_idx = tl.program_id(1)
    blk_idx = tl.program_id(2)

    block_start = blk_idx * BLOCK_T

    # seq_lens may be int32 or int64; load as int64 for the comparisons below.
    seq_len = tl.load(seq_lens_ptr + bs_idx).to(tl.int64)
    history_len = seq_len - 1  # exclude current decode position

    s_offsets = tl.arange(0, S)  # [S]
    chans = tl.load(channel_idx_ptr + kv_idx * chan_stride_h + s_offsets).to(tl.int64)

    # ---- Inline Q_label gather + GQA reduction. ----
    if GQA_REDUCTION == 0:  # max_abs: keep value with max |abs| per channel
        q_label = tl.zeros([S], dtype=tl.float32)
        max_abs = tl.zeros([S], dtype=tl.float32)
        for g in tl.static_range(GROUP):
            qh = kv_idx * GROUP + g
            q_offs = bs_idx * Q_stride_b + qh * Q_stride_h + chans
            q_vals = tl.load(Q_ptr + q_offs).to(tl.float32)
            absvals = tl.abs(q_vals)
            update = absvals > max_abs
            q_label = tl.where(update, q_vals, q_label)
            max_abs = tl.where(update, absvals, max_abs)
    else:
        # mean / soq share the running-sum accumulators
        sum_q = tl.zeros([S], dtype=tl.float32)
        sum_sq = tl.zeros([S], dtype=tl.float32)
        for g in tl.static_range(GROUP):
            qh = kv_idx * GROUP + g
            q_offs = bs_idx * Q_stride_b + qh * Q_stride_h + chans
            q_vals = tl.load(Q_ptr + q_offs).to(tl.float32)
            sum_q += q_vals
            sum_sq += q_vals * q_vals
        if GQA_REDUCTION == 1:  # mean
            q_label = sum_q / GROUP
        else:  # soq: sign(mean) * sqrt(sum_sq)
            sign = tl.where(sum_q >= 0.0, 1.0, -1.0)
            q_label = sign * tl.sqrt(sum_sq)

    # ---- Score the block. ----
    t_offsets = block_start + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    valid = t_offsets < history_len  # [BLOCK_T] bool

    rpi = tl.load(req_pool_indices_ptr + bs_idx).to(tl.int64)
    r2t_offs = rpi * r2t_stride_r + t_offsets
    phys = tl.load(req_to_token_ptr + r2t_offs, mask=valid, other=0).to(tl.int64)

    kl_offs = phys[:, None] * kl_stride_t + kv_idx * kl_stride_h + s_offsets[None, :]
    k_label_tile = tl.load(k_label_ptr + kl_offs, mask=valid[:, None], other=0.0).to(
        tl.float32
    )  # [BLOCK_T, S]

    scores = tl.sum(k_label_tile * q_label[None, :], axis=1)  # [BLOCK_T] fp32
    scores = tl.where(valid, scores, NEG_INF)

    # ---- Iterative top-K_BLOCK. ----
    # Repeat K_BLOCK times: take argmax (with min-position tie-break),
    # write to output, mask out the chosen position.
    positions = tl.arange(0, BLOCK_T).to(tl.int32)  # [BLOCK_T]

    out_logical = tl.zeros([K_BLOCK], dtype=tl.int32) - 1  # init to -1
    out_scores = tl.zeros([K_BLOCK], dtype=tl.float32) + NEG_INF

    # We need to write per-iteration to out_logical[k] / out_scores[k].
    # Triton in-register tensors are immutable; we use `tl.where` over a
    # one-hot k_index to update.
    out_k_range = tl.arange(0, K_BLOCK).to(tl.int32)

    for k in tl.static_range(K_BLOCK):
        max_score = tl.max(scores, axis=0)  # scalar fp32
        # Min-position tie-break: among positions where score == max,
        # pick the smallest. This matches the torch reference's
        # argsort(stable=True).
        is_max = scores == max_score
        pos_or_inf = tl.where(is_max, positions, BLOCK_T)
        max_pos = tl.min(pos_or_inf, axis=0)  # scalar int32
        logical_int64 = block_start + max_pos.to(tl.int64)
        logical_valid = logical_int64.to(tl.int32)
        # If the best remaining score is NEG_INF (all positions in this
        # block are past history or already picked), emit the -1 / NEG_INF
        # sentinel pair instead of a valid-looking logical position.
        # `max_score > NEG_INF` is the gate; we use a slightly looser
        # threshold to avoid float-bit-equality issues.
        slot_active = max_score > -1.0e37
        logical = tl.where(slot_active, logical_valid, -1)
        score_to_write = tl.where(slot_active, max_score, NEG_INF)

        # Update register tensors at index k.
        is_kth = out_k_range == k
        out_logical = tl.where(is_kth, logical, out_logical)
        out_scores = tl.where(is_kth, score_to_write, out_scores)

        # Mask out the chosen position so the next iteration picks
        # something else (only matters when we actually picked one).
        scores = tl.where(positions == max_pos, NEG_INF, scores)

    # ---- Write outputs. ----
    out_base_log = (
        bs_idx * btl_stride_b + kv_idx * btl_stride_h + blk_idx * btl_stride_blk
    )
    out_base_scr = (
        bs_idx * bts_stride_b + kv_idx * bts_stride_h + blk_idx * bts_stride_blk
    )
    tl.store(block_topk_logical_ptr + out_base_log + out_k_range, out_logical)
    tl.store(block_topk_scores_ptr + out_base_scr + out_k_range, out_scores)


def ds_select_stage1_block_topk(
    *,
    queries: torch.Tensor,  # [bs, H_q, D]
    channel_idx: torch.Tensor,  # [H_kv, S] int32
    k_label: torch.Tensor,  # [T, H_kv, S]
    req_to_token: torch.Tensor,  # [max_reqs, max_ctx]
    req_pool_indices: torch.Tensor,  # [bs] int64
    seq_lens: torch.Tensor,  # [bs] int64
    num_kv_heads: int,
    block_t: int,
    k_block: int,
    gqa_reduction_id: int,
    block_topk_logical: torch.Tensor = None,  # OUT, optional preallocated
    block_topk_scores: torch.Tensor = None,  # OUT, optional preallocated
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA wrapper for the stage-1 block-topk Triton kernel.

    Outputs (per program: one (bs, kv_head, block) cell):
      * `block_topk_logical[bs, H_kv, num_blocks, K_block]` int32 logical
        token positions; -1 for slots in blocks beyond history.
      * `block_topk_scores[bs, H_kv, num_blocks, K_block]` fp32 scores;
        NEG_INF for slots in blocks beyond history.

    Caller can pass preallocated output buffers (production wiring
    allocates these once at algorithm init). Otherwise allocates them.
    """
    if not queries.is_cuda:
        raise RuntimeError(
            "ds_select_stage1_block_topk requires CUDA tensors; use "
            "ds_select_stage1_block_topk_torch_ref for CPU testing."
        )

    bs, h_q, d = queries.shape
    h_kv = num_kv_heads
    if h_q % h_kv != 0:
        raise ValueError(f"H_q={h_q} not divisible by H_kv={h_kv}")
    group = h_q // h_kv
    s = channel_idx.shape[1]
    max_ctx = req_to_token.shape[1]
    num_blocks = (max_ctx + block_t - 1) // block_t

    if block_topk_logical is None:
        block_topk_logical = torch.full(
            (bs, h_kv, num_blocks, k_block),
            -1,
            dtype=torch.int32,
            device=queries.device,
        )
    if block_topk_scores is None:
        block_topk_scores = torch.full(
            (bs, h_kv, num_blocks, k_block),
            float("-inf"),
            dtype=torch.float32,
            device=queries.device,
        )

    grid = (bs, h_kv, num_blocks)
    _ds_select_stage1_block_topk_kernel[grid](
        queries,
        channel_idx,
        k_label,
        req_to_token,
        req_pool_indices,
        seq_lens,
        block_topk_logical,
        block_topk_scores,
        queries.stride(0),
        queries.stride(1),
        channel_idx.stride(0),
        k_label.stride(0),
        k_label.stride(1),
        req_to_token.stride(0),
        block_topk_logical.stride(0),
        block_topk_logical.stride(1),
        block_topk_logical.stride(2),
        block_topk_scores.stride(0),
        block_topk_scores.stride(1),
        block_topk_scores.stride(2),
        H_q=h_q,
        H_kv=h_kv,
        GROUP=group,
        D=d,
        S=s,
        BLOCK_T=block_t,
        K_BLOCK=k_block,
        GQA_REDUCTION=gqa_reduction_id,
    )
    return block_topk_logical, block_topk_scores


# --------------------------------------------------------------------------- #
# Stage 2 — merge per (bs, kv_head): top-effective_budget by score.           #
# --------------------------------------------------------------------------- #


def ds_select_stage2_merge_torch_ref(
    *,
    block_topk_logical: torch.Tensor,  # [bs, H_kv, num_blocks, K_BLOCK] int32
    block_topk_scores: torch.Tensor,  # [bs, H_kv, num_blocks, K_BLOCK] fp32
    effective_budget: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference impl for stage-2 merge.

    Per (bs, kv_head): collect all `num_blocks * K_BLOCK` candidates,
    sort by score descending, take top `effective_budget`. Slots beyond
    the active-candidate count emit (-1, NEG_INF).

    Returns:
        merged_logical: `[bs, H_kv, effective_budget]` int32.
        merged_scores:  `[bs, H_kv, effective_budget]` fp32.
    """
    bs, h_kv, num_blocks, k_block = block_topk_logical.shape
    device = block_topk_logical.device

    out_logical = torch.full(
        (bs, h_kv, effective_budget), -1, dtype=torch.int32, device=device
    )
    out_scores = torch.full(
        (bs, h_kv, effective_budget),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )

    flat_log = block_topk_logical.reshape(bs, h_kv, num_blocks * k_block)
    flat_scr = block_topk_scores.reshape(bs, h_kv, num_blocks * k_block)
    # Sort by score descending; ties broken by smaller logical position
    # so the result is deterministic (matches the kernel's reduction).
    for b in range(bs):
        for h in range(h_kv):
            scores = flat_scr[b, h].clone()
            logicals = flat_log[b, h].clone()
            # Stable sort by score descending, then by logical ascending
            # (achieved by sorting on a composite key).
            order = torch.argsort(scores, descending=True, stable=True)
            top = order[:effective_budget]
            n_valid = effective_budget
            sel_log = logicals[top]
            sel_scr = scores[top]
            # Mark slots beyond the count of active (score > NEG_INF and
            # logical >= 0) candidates as sentinels.
            valid_mask = (sel_log >= 0) & (sel_scr > -1e30)
            out_logical[b, h, :n_valid] = torch.where(
                valid_mask, sel_log, torch.full_like(sel_log, -1)
            )
            out_scores[b, h, :n_valid] = torch.where(
                valid_mask, sel_scr, torch.full_like(sel_scr, float("-inf"))
            )
    return out_logical, out_scores


@triton.jit
def _ds_select_stage2_merge_kernel(
    block_topk_logical_ptr,
    block_topk_scores_ptr,
    merged_logical_ptr,
    merged_scores_ptr,
    # block_topk strides ([bs, H_kv, num_blocks, K_BLOCK]; last contig)
    btl_stride_b,
    btl_stride_h,
    bts_stride_b,
    bts_stride_h,
    # merged strides ([bs, H_kv, EFFECTIVE_BUDGET]; last contig)
    ml_stride_b,
    ml_stride_h,
    ms_stride_b,
    ms_stride_h,
    # constexprs
    NUM_CANDIDATES: tl.constexpr,  # num_blocks * K_BLOCK (must be <= merge_safe_threshold)
    NUM_CANDIDATES_PADDED: tl.constexpr,  # next-pow2(NUM_CANDIDATES) for tl.arange
    EFFECTIVE_BUDGET: tl.constexpr,
    EFFECTIVE_BUDGET_PADDED: tl.constexpr,  # next-pow2(EFFECTIVE_BUDGET) for tl.arange
):
    """One program per (bs, kv_head). In-program iterative argmax for
    top-EFFECTIVE_BUDGET candidates by score.

    Caller guarantees `NUM_CANDIDATES <= merge_safe_threshold` (default
    4096) — the chunked merge path is a v1.1.x escape hatch we don't
    need until profiling demands it.

    `tl.arange` requires a power-of-2 size, so the kernel works on
    `NUM_CANDIDATES_PADDED` (next pow2) and masks the padding tail to
    NEG_INF / -1.
    """
    NEG_INF = -1e38

    bs_idx = tl.program_id(0)
    kv_idx = tl.program_id(1)

    cand_offsets = tl.arange(0, NUM_CANDIDATES_PADDED)  # [NUM_CANDIDATES_PADDED]
    in_range = cand_offsets < NUM_CANDIDATES
    btl_offs = bs_idx * btl_stride_b + kv_idx * btl_stride_h + cand_offsets
    bts_offs = bs_idx * bts_stride_b + kv_idx * bts_stride_h + cand_offsets

    # Mask out-of-range tail loads; treat as sentinel.
    logicals = tl.load(block_topk_logical_ptr + btl_offs, mask=in_range, other=-1)
    scores = tl.load(block_topk_scores_ptr + bts_offs, mask=in_range, other=NEG_INF).to(
        tl.float32
    )
    # Drop sentinel candidates (-1 logical → unscored slot, plus padding).
    valid = (logicals >= 0) & in_range
    scores = tl.where(valid, scores, NEG_INF)

    out_k_range = tl.arange(0, EFFECTIVE_BUDGET_PADDED).to(tl.int32)
    out_in_range = out_k_range < EFFECTIVE_BUDGET
    out_logical = tl.zeros([EFFECTIVE_BUDGET_PADDED], dtype=tl.int32) - 1
    out_scores = tl.zeros([EFFECTIVE_BUDGET_PADDED], dtype=tl.float32) + NEG_INF

    for k in tl.static_range(EFFECTIVE_BUDGET):
        max_score = tl.max(scores, axis=0)
        # Min-position tie-break (matches torch ref's stable sort).
        is_max = scores == max_score
        pos_or_inf = tl.where(is_max, cand_offsets, NUM_CANDIDATES_PADDED)
        max_pos = tl.min(pos_or_inf, axis=0)

        slot_active = max_score > -1.0e37
        # Gather logical at max_pos. Triton can't do dynamic indexing into
        # a register tensor, so use mask-and-reduce: pick the `logicals`
        # value where position == max_pos.
        is_picked = cand_offsets == max_pos
        picked_logical = tl.sum(tl.where(is_picked, logicals, 0), axis=0).to(tl.int32)
        logical = tl.where(slot_active, picked_logical, -1)
        score_to_write = tl.where(slot_active, max_score, NEG_INF)

        is_kth = out_k_range == k
        out_logical = tl.where(is_kth, logical, out_logical)
        out_scores = tl.where(is_kth, score_to_write, out_scores)

        # Mask out the chosen position so the next iteration moves on.
        scores = tl.where(cand_offsets == max_pos, NEG_INF, scores)

    out_log_base = bs_idx * ml_stride_b + kv_idx * ml_stride_h
    out_scr_base = bs_idx * ms_stride_b + kv_idx * ms_stride_h
    tl.store(
        merged_logical_ptr + out_log_base + out_k_range,
        out_logical,
        mask=out_in_range,
    )
    tl.store(
        merged_scores_ptr + out_scr_base + out_k_range,
        out_scores,
        mask=out_in_range,
    )


def ds_select_stage2_merge(
    *,
    block_topk_logical: torch.Tensor,
    block_topk_scores: torch.Tensor,
    effective_budget: int,
    merge_safe_threshold: int = 4096,
    merged_logical: torch.Tensor = None,
    merged_scores: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA wrapper for the stage-2 merge Triton kernel.

    Caller (v1.1-7 wiring) must ensure `num_blocks * K_BLOCK <=
    merge_safe_threshold` (default 4096) — this is checked by
    `DoubleSparsityRuntimeConfig.warn_capacity` at startup. The chunked
    merge path is a v1.1.x escape hatch when that constraint is
    violated; until profiling demands it, we hard-error here.
    """
    if not block_topk_logical.is_cuda:
        raise RuntimeError("ds_select_stage2_merge requires CUDA tensors")
    bs, h_kv, num_blocks, k_block = block_topk_logical.shape
    num_candidates = num_blocks * k_block
    if num_candidates > merge_safe_threshold:
        raise RuntimeError(
            f"stage-2 merge candidates {num_candidates} = {num_blocks} * {k_block} "
            f"exceeds merge_safe_threshold={merge_safe_threshold}. Lower k_block, "
            f"raise block_t, or wire the chunked merge path (v1.1.x)."
        )
    if effective_budget > num_candidates:
        raise ValueError(
            f"effective_budget={effective_budget} > num_candidates={num_candidates}; "
            f"caller must clamp via min(token_budget, num_blocks * k_block)"
        )

    if merged_logical is None:
        merged_logical = torch.full(
            (bs, h_kv, effective_budget),
            -1,
            dtype=torch.int32,
            device=block_topk_logical.device,
        )
    if merged_scores is None:
        merged_scores = torch.full(
            (bs, h_kv, effective_budget),
            float("-inf"),
            dtype=torch.float32,
            device=block_topk_logical.device,
        )

    grid = (bs, h_kv)
    _ds_select_stage2_merge_kernel[grid](
        block_topk_logical,
        block_topk_scores,
        merged_logical,
        merged_scores,
        block_topk_logical.stride(0),
        block_topk_logical.stride(1),
        block_topk_scores.stride(0),
        block_topk_scores.stride(1),
        merged_logical.stride(0),
        merged_logical.stride(1),
        merged_scores.stride(0),
        merged_scores.stride(1),
        NUM_CANDIDATES=num_candidates,
        NUM_CANDIDATES_PADDED=triton.next_power_of_2(num_candidates),
        EFFECTIVE_BUDGET=effective_budget,
        EFFECTIVE_BUDGET_PADDED=triton.next_power_of_2(effective_budget),
    )
    return merged_logical, merged_scores


# --------------------------------------------------------------------------- #
# Score-aware union per batch — Triton, capture-safe, no [bs, max_ctx] alloc. #
# --------------------------------------------------------------------------- #


def ds_union_per_batch_torch_ref(
    *,
    merged_logical: torch.Tensor,  # [bs, H_kv, effective_budget] int32
    merged_scores: torch.Tensor,  # [bs, H_kv, effective_budget] fp32
    seq_lens: torch.Tensor,  # [bs] int64
    sink_tokens: int,
    recent_tokens: int,
    min_seq_len: int,
    max_selected_per_request: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference impl of the score-aware union pass.

    Per batch row:
      1. Always-keep: sink window `[0, sink_tokens)` ∪ recency window
         `[seq - recent_tokens, seq)`. Bypass scoring entirely.
      2. History candidates: union of `merged_logical[b, :, :]` across
         kv_heads. Each candidate carries its score; if the same logical
         token appears in multiple kv_heads, keep the **max** score.
      3. Cap by score: if `len(history) + len(always_keep) >
         max_selected_per_request`, drop the **lowest-score** history
         candidates first (NEVER drop by logical position).
      4. Dense fallback (`seq < min_seq_len`): replace whole row with
         `[0, seq)` (capped at max_selected_per_request).
      5. Final output: sort surviving set in **logical** order
         (FA3 wants logical-order page tables, not score-sorted).

    Returns:
      selected_logical: [bs, max_selected_per_request] int32 (-1 padding).
      valid_lengths:    [bs] int32.
    """
    bs, h_kv, _ = merged_logical.shape
    device = merged_logical.device
    msel = max_selected_per_request

    out_logical = torch.full((bs, msel), -1, dtype=torch.int32, device=device)
    out_valid = torch.zeros(bs, dtype=torch.int32, device=device)

    for b in range(bs):
        seq_len = int(seq_lens[b].item())

        # Dense fallback
        if seq_len < min_seq_len:
            n = min(seq_len, msel)
            out_logical[b, :n] = torch.arange(n, dtype=torch.int32, device=device)
            out_valid[b] = n
            continue

        # Always-keep set: sink + recency
        sink_n = max(min(sink_tokens, seq_len), 0)
        rec_lo = max(seq_len - recent_tokens, 0)
        rec_hi = seq_len
        always_keep = sorted(set(range(0, sink_n)) | set(range(rec_lo, rec_hi)))

        # History candidates: dedup by logical, score = max across kv_heads.
        # Skip sentinels (-1) and any candidates that fall in always_keep
        # (those are guaranteed to land in the output anyway).
        always_keep_set = set(always_keep)
        history_score: dict[int, float] = {}
        for h in range(h_kv):
            for k in range(merged_logical.shape[2]):
                logical = int(merged_logical[b, h, k].item())
                score = float(merged_scores[b, h, k].item())
                if logical < 0 or score <= -1e30:
                    continue
                if logical in always_keep_set:
                    continue  # already kept; don't double-count
                if logical not in history_score or score > history_score[logical]:
                    history_score[logical] = score

        # Cap by score: drop lowest-score history first.
        history_capacity = msel - len(always_keep)
        if history_capacity < 0:
            # always_keep alone exceeds cap; truncate (rare)
            survivors = always_keep[:msel]
        else:
            history = sorted(history_score.items(), key=lambda kv: -kv[1])[
                :history_capacity
            ]
            survivors = sorted(always_keep + [logical for logical, _ in history])

        n = min(len(survivors), msel)
        out_logical[b, :n] = torch.tensor(
            survivors[:n], dtype=torch.int32, device=device
        )
        out_valid[b] = n

    return out_logical, out_valid


def ds_union_per_batch(
    *,
    merged_logical: torch.Tensor,  # [bs, H_kv, effective_budget] int32
    merged_scores: torch.Tensor,  # [bs, H_kv, effective_budget] fp32
    seq_lens: torch.Tensor,  # [bs] int64
    sink_tokens: int,
    recent_tokens: int,
    min_seq_len: int,
    max_selected_per_request: int,
    union_safe_threshold: int = 4096,
    selected_logical: torch.Tensor = None,
    valid_lengths: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Score-aware union pass — bounded-shape torch CUDA implementation.

    Capture-safe by virtue of:
      * No tensor scales with `max_ctx`. All scratch is bounded by
        `H_kv * effective_budget + sink + recent + max_selected`,
        well under 8K total in production configs.
      * Output tensors (`selected_logical`, `valid_lengths`) are
        preallocated by the caller; we mutate via `.copy_()` and
        `index_copy_` only. No torch.unique, no allocate-then-rebind.
      * `torch.topk` and `torch.sort` on small fixed-shape inputs are
        capture-safe (cuBLAS / cub workspaces, no host syncs).

    Score-aware policy (matches torch ref):
      1. Always-keep set: sink ∪ recency window, dedup'd.
      2. History candidates: dedup by logical (max score across
         kv_heads), drop those that overlap always-keep, mask sentinels.
      3. Top-(max_selected - len(always_keep)) by score from history.
      4. Combine + sort ascending by logical position.
      5. Dense fallback (`seq < min_seq_len`): replace whole row with
         `[0, seq)` capped at max_selected.

    The "Triton or torch" choice in the v1.1 plan is settled by the
    capture-safety constraint — a Triton implementation of this same
    algorithm offers no clear win at these shapes, and a torch impl
    is far easier to verify. v1.1.x can swap if profiling demands.
    """
    if not merged_logical.is_cuda:
        raise RuntimeError("ds_union_per_batch requires CUDA tensors")
    bs, h_kv, effective_budget = merged_logical.shape
    num_history_candidates = h_kv * effective_budget
    if num_history_candidates > union_safe_threshold:
        raise RuntimeError(
            f"union candidates {num_history_candidates} = {h_kv} * {effective_budget} "
            f"exceeds union_safe_threshold={union_safe_threshold}. Lower token_budget, "
            f"or wire the chunked union path (v1.1.x)."
        )

    device = merged_logical.device
    msel = max_selected_per_request

    if selected_logical is None:
        selected_logical = torch.full((bs, msel), -1, dtype=torch.int32, device=device)
    if valid_lengths is None:
        valid_lengths = torch.zeros(bs, dtype=torch.int32, device=device)

    NEG_INF = float("-inf")

    # Flatten history candidates per row: [bs, h_kv * effective_budget].
    flat_log = merged_logical.reshape(bs, num_history_candidates).to(torch.int32)
    flat_scr = merged_scores.reshape(bs, num_history_candidates).to(torch.float32)

    # Mask sentinels.
    valid_mask = (flat_log >= 0) & (flat_scr > -1e30)
    flat_scr = torch.where(valid_mask, flat_scr, torch.full_like(flat_scr, NEG_INF))
    flat_log = torch.where(valid_mask, flat_log, torch.full_like(flat_log, -1))

    # Build always-keep ranges per row (sink + recency, clamped to [0, seq)).
    seq_lens_dev = seq_lens.to(device).to(torch.int64)
    # Sink: positions [0, min(sink, seq))
    # Recency: positions [max(seq - recent, 0), seq)
    # We compute fixed-shape masks: a [bs, max_keep_count] tensor where
    # invalid slots become -1.

    # Mask history candidates that fall in always-keep windows (drop to
    # NEG_INF so we don't double-count).
    pos_in_history = flat_log.to(torch.int64)  # [bs, NUM_HIST] int64
    in_sink = (pos_in_history < sink_tokens) & (pos_in_history >= 0)
    rec_lo = (seq_lens_dev - recent_tokens).clamp_min(0).unsqueeze(1)  # [bs, 1]
    in_recent = (pos_in_history >= rec_lo) & (
        pos_in_history < seq_lens_dev.unsqueeze(1)
    )
    in_always = in_sink | in_recent
    flat_scr = torch.where(in_always, torch.full_like(flat_scr, NEG_INF), flat_scr)

    # Dedup by logical: keep ONE entry per (b, logical) with the max
    # score across kv-heads. Achieved by a two-key stable sort —
    # (1) by score descending (stable), (2) by logical ascending
    # (stable). After step (2), entries with the same logical are
    # adjacent AND the max-score representative is the FIRST in each
    # run (because step (1) put it at the top of every score-tier
    # before step (2) preserved that order within equal-logical groups).
    # Then any position with `same_as_prev` is a duplicate to mask.
    #
    # The naive single-key sort + adjacent-pair-mask is wrong for runs
    # of length ≥3 with non-monotone scores (e.g. logical=7 with
    # [10, 5, 9] would leave both 10 and 9 alive — confirmed by the
    # `TestUnionDedupCorrectness` fuzzer).
    desc_perm = (-flat_scr).argsort(dim=1, stable=True)
    flat_log_d = torch.gather(flat_log, 1, desc_perm)
    flat_scr_d = torch.gather(flat_scr, 1, desc_perm)
    asc_perm = flat_log_d.argsort(dim=1, stable=True)
    sorted_log = torch.gather(flat_log_d, 1, asc_perm)
    sorted_scr = torch.gather(flat_scr_d, 1, asc_perm)
    same_as_prev = torch.zeros_like(sorted_log, dtype=torch.bool)
    same_as_prev[:, 1:] = sorted_log[:, 1:] == sorted_log[:, :-1]
    sorted_scr = torch.where(
        same_as_prev, torch.full_like(sorted_scr, NEG_INF), sorted_scr
    )

    # Top-`history_capacity` by score.
    history_capacity = msel - sink_tokens - recent_tokens
    history_capacity = max(history_capacity, 0)
    if history_capacity > 0 and num_history_candidates > 0:
        k_hist = min(history_capacity, num_history_candidates)
        topk_idx = sorted_scr.topk(k_hist, dim=1, sorted=False).indices
        # Gather logicals at the topk positions
        topk_log = torch.gather(sorted_log, 1, topk_idx)
        topk_scr = torch.gather(sorted_scr, 1, topk_idx)
        # Mark slots where score is NEG_INF as sentinels
        topk_log = torch.where(
            topk_scr > -1e30, topk_log, torch.full_like(topk_log, -1)
        )
    else:
        topk_log = torch.empty(bs, 0, dtype=torch.int32, device=device)

    # Build always-keep tensor: [bs, sink + recent].
    keep_count = sink_tokens + recent_tokens
    if keep_count > 0:
        keep = torch.full((bs, keep_count), -1, dtype=torch.int32, device=device)
        # Sink slots [0, sink_tokens), masked by < seq.
        if sink_tokens > 0:
            sink_range = torch.arange(sink_tokens, device=device, dtype=torch.int64)
            sink_valid = sink_range < seq_lens_dev.unsqueeze(1)
            keep[:, :sink_tokens] = torch.where(
                sink_valid,
                sink_range.expand(bs, -1).to(torch.int32),
                torch.full((bs, sink_tokens), -1, dtype=torch.int32, device=device),
            )
        # Recency slots [seq - recent_tokens, seq); skip overlap with sink.
        if recent_tokens > 0:
            rec_offset = torch.arange(recent_tokens, device=device, dtype=torch.int64)
            rec_pos = (seq_lens_dev - recent_tokens).unsqueeze(1) + rec_offset
            rec_valid = (
                (rec_pos >= 0)
                & (rec_pos < seq_lens_dev.unsqueeze(1))
                & (rec_pos >= sink_tokens)  # avoid overlap with sink
            )
            keep[:, sink_tokens : sink_tokens + recent_tokens] = torch.where(
                rec_valid,
                rec_pos.to(torch.int32),
                torch.full(rec_pos.shape, -1, dtype=torch.int32, device=device),
            )
    else:
        keep = torch.empty(bs, 0, dtype=torch.int32, device=device)

    # Combine: [bs, keep_count + history_capacity]; sentinel-mark, then
    # sort logical-ascending (sentinels go to end via large +inf cast).
    combined = torch.cat([keep, topk_log], dim=1)  # [bs, total]
    pos_or_inf = torch.where(
        combined >= 0,
        combined.to(torch.float32),
        torch.full_like(combined, 2.0e9, dtype=torch.float32),
    )
    sorted_marks, _ = pos_or_inf.sort(dim=1)
    # Take the first msel slots; sentinel slots are 2e9 → -1.
    take = (
        sorted_marks[:, :msel]
        if sorted_marks.shape[1] >= msel
        else torch.cat(
            [
                sorted_marks,
                torch.full(
                    (bs, msel - sorted_marks.shape[1]),
                    2.0e9,
                    dtype=torch.float32,
                    device=device,
                ),
            ],
            dim=1,
        )
    )
    final_logical = torch.where(
        take < 1e9, take.to(torch.int32), torch.full_like(take, -1, dtype=torch.int32)
    )

    # Dense fallback: rows where seq < min_seq_len get [0, seq) padded -1.
    df_mask = seq_lens_dev < min_seq_len  # [bs] bool
    df_range = torch.arange(msel, device=device, dtype=torch.int64).unsqueeze(0)
    df_valid = df_range < seq_lens_dev.unsqueeze(1)
    df_logical = torch.where(
        df_valid,
        df_range.expand(bs, -1).to(torch.int32),
        torch.full((bs, msel), -1, dtype=torch.int32, device=device),
    )
    final_logical = torch.where(df_mask.unsqueeze(1), df_logical, final_logical)

    # Write outputs in-place.
    selected_logical.copy_(final_logical)
    valid_lengths.copy_((final_logical >= 0).sum(dim=1).to(torch.int32))
    return selected_logical, valid_lengths


__all__ = [
    "ds_select_stage1_block_topk",
    "ds_select_stage1_block_topk_torch_ref",
    "ds_select_stage2_merge",
    "ds_select_stage2_merge_torch_ref",
    "ds_union_per_batch",
    "ds_union_per_batch_torch_ref",
]
