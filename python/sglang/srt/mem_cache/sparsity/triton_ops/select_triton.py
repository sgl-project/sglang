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


__all__ = [
    "ds_select_stage1_block_topk",
    "ds_select_stage1_block_topk_torch_ref",
]
