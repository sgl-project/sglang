"""NPU varlen multi-token sparse-attention PREFILL kernels.

Batch block-scoring across a *query block* of ``BLOCK_SIZE_Q`` tokens,
mirroring the GPU ``prefill/`` varlen kernels so the seq dimension stays
parallelized. Extends the NPU decode score kernel: Q grows from ``[H, D]`` to
``[BLOCK_SIZE_Q * H, D]`` and the QK dot stays 2D (a 3D reshape is
Ascend-TBE-unsafe). Host precomputes per-query-block varlen mappings.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.flash_block_score_decode import (
    _choose_num_score_chunks,
    _next_power_of_2,
)

# Fused post-score TopK (flash_prefill_bnsd_topk_from_score) is only used up to
# this many KV blocks (32K context at block_size=128); see the gate in
# flash_prefill_bnsd_indexer / flash_prefill_bnsd_with_topk_idx.

# Tunable launch configs for the score kernels. Bench/tuning scripts override
# these module globals before the first launch to sweep configs (NOT
# triton.autotune, so each shape compiles to one deterministic artifact --
# cuda-graph safe). Mirrors topk_sparse_decode.py.
_SCORE_NW = 4
_SCORE_NS = 2
# num_warps does not affect numerics.
_SCORE_ATTN_NW = 8
_SCORE_ATTN_NS = 2


@triton.jit
def _prefill_bnsd_score_kernel(
    q_ptr,  # [total_q, num_idx_heads, head_dim]
    k_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_dim]
    block_table_ptr,  # [all_seqblock_q, max_num_blocks]
    qb_to_qstart_ptr,  # [all_seqblock_q]
    qb_to_qblock_ptr,  # [all_seqblock_q]
    qb_seq_lens_ptr,  # [all_seqblock_q]
    qb_qend_ptr,  # [all_seqblock_q] exclusive q upper bound (cu_seqlens[r+1])
    score_ptr,  # [num_idx_heads, total_q, max_seqblock_k]
    # scalars
    total_q,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    all_seqblock_q,
    num_score_chunks,
    sm_scale,
    # strides
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_bt_q,
    stride_bt_n,
    stride_s_h,
    stride_s_q,
    stride_s_n,
    # constexpr meta
    block_size: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    # Number of consecutive logical KV blocks fused into one tl.dot per step.
    # 1 == single-block path; 2/4 grow the dot's N dimension. Power of two so
    # BLOCK_SIZE_N = BLOCKS_PER_STEP * block_size stays aligned.
    BLOCKS_PER_STEP: tl.constexpr,
):
    """Score one query-block x one kv_head tile.

    Grid ``(all_seqblock_q * num_score_chunks, num_kv_heads)``: ``pid_qbc % all_seqblock_q`` -> query-block, ``//`` -> a chunk of consecutive KV blocks.
    """
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_qbc = tl.program_id(0)
    pid_kh = tl.program_id(1)

    pid_qb = pid_qbc % all_seqblock_q
    pid_c = pid_qbc // all_seqblock_q

    seq_len = tl.load(qb_seq_lens_ptr + pid_qb).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)
    chunk_size_blocks = tl.maximum(1, tl.cdiv(num_blocks, num_score_chunks))
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return

    q_start = tl.load(qb_to_qstart_ptr + pid_qb).to(tl.int32)
    q_block_local = tl.load(qb_to_qblock_ptr + pid_qb).to(tl.int32)
    q_end = tl.load(qb_qend_ptr + pid_qb).to(tl.int32)

    off_d = tl.arange(0, BLOCK_SIZE_D)  # [D]
    off_n = tl.arange(0, BLOCK_SIZE_N)  # [N]
    # Flattened (query-row, head-within-group) -> [BSQ * H], kept 1D: a [BSQ, H]
    # int32 2D tensor with a small H fails the Ascend TBE stride-alignment check
    # ("cannot align 0 axis"), so build the flat indices directly via divmod.
    off_qh = tl.arange(0, BLOCK_SIZE_Q * BLOCK_SIZE_H)  # [BSQ*H]
    qi = off_qh // BLOCK_SIZE_H  # query row within the block
    hh = off_qh % BLOCK_SIZE_H  # head index within the GQA group
    pid_h_base = pid_kh * gqa_group_size
    q_token_raw = q_start + q_block_local * BLOCK_SIZE_Q + qi  # [BSQ*H]
    head_flat = pid_h_base + hh  # actual idx-head index per row
    # q_end (= cu_seqlens[r+1]) bounds to THIS request's real tokens so phantom
    # rows of a partial-tail q-block (block_size_q>1, extend not a BLOCK_SIZE_Q
    # multiple) are masked. Clamp q_token for the Q load to [q_start, q_end-1] so
    # phantom rows never read into the next request's Q data. q_end <= total_q.
    row_valid = (q_token_raw < q_end) & (hh < gqa_group_size)
    q_token_flat = tl.maximum(q_start, tl.minimum(q_token_raw, q_end - 1))

    # Q load: [BSQ*H, D]  (clamped q_token keeps reads in-request)
    q_offsets = (
        q_token_flat[:, None] * stride_q_n
        + head_flat[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=row_valid[:, None] & (off_d[None, :] < head_dim),
        other=0.0,
    )

    sm_scale_log2e = sm_scale * 1.4426950409

    if BLOCKS_PER_STEP == 1:
        # Original single-block path: one KV block (block_size tokens) per dot.
        num_steps = chunk_end_block - chunk_start_block
        for step in tl.range(num_steps):
            logical_block = chunk_start_block + step
            physical_block = tl.load(
                block_table_ptr + pid_qb * stride_bt_q + logical_block * stride_bt_n
            ).to(tl.int64)

            key_pos = logical_block * block_size + off_n  # [N]
            pos_mask = key_pos < seq_len

            k_offsets = (
                physical_block * stride_k_block
                + off_n[None, :] * stride_k_offset
                + pid_kh * stride_k_h
                + off_d[:, None] * stride_k_d
            )
            k = tl.load(
                k_cache_ptr + k_offsets,
                mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
                other=0.0,
            )

            qk = tl.dot(q, k) * sm_scale_log2e  # [BSQ*H, N], 2D dot -- no 3D reshape

            # Causal: query token >= key position (use q_token_raw for correct per-
            # token position; phantom rows are masked by row_valid downstream).
            causal = q_token_raw[:, None] >= key_pos[None, :]
            qk = tl.where(causal & pos_mask[None, :], qk, float("-inf"))

            sub_max = tl.max(qk, axis=1)  # [BSQ*H]
            if SCORE_TYPE == "max":
                score = sub_max
            else:
                score = sub_max + tl.log2(
                    tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1)
                )
                score = tl.where(score != score, float("-inf"), score)

            s_offsets = (
                head_flat * stride_s_h
                + q_token_raw * stride_s_q
                + logical_block * stride_s_n
            )
            tl.store(
                score_ptr + s_offsets,
                score.to(score_ptr.dtype.element_ty),
                mask=row_valid,
            )
    else:
        # Multi-block path: BLOCKS_PER_STEP consecutive logical KV blocks are
        # gathered into one [D, BLOCKS_PER_STEP*block_size] K tile and scored with
        # a single larger tl.dot. No 3D reshape (Ascend TBE rejects it); per-block
        # scores are extracted with a column-select mask + max.
        sub_id = off_n // block_size  # [BLOCK_SIZE_N] which sub-block in [0, BPS)
        inn = off_n % block_size  # [BLOCK_SIZE_N] offset within the block
        num_steps = tl.cdiv(chunk_end_block - chunk_start_block, BLOCKS_PER_STEP)
        for step in tl.range(num_steps):
            logical_base = chunk_start_block + step * BLOCKS_PER_STEP
            logical_block_n = logical_base + sub_id  # [BLOCK_SIZE_N]
            # Clamp the block-table index to a valid block so the gather never
            # reads out of bounds; out-of-range sub-blocks are masked away below.
            logical_block_load = tl.minimum(logical_block_n, chunk_end_block - 1)
            physical_block_n = tl.load(
                block_table_ptr
                + pid_qb * stride_bt_q
                + logical_block_load * stride_bt_n
            ).to(tl.int64)

            key_pos = logical_base * block_size + off_n  # [BLOCK_SIZE_N]
            pos_mask = key_pos < seq_len

            k_offsets = (
                physical_block_n[None, :] * stride_k_block
                + inn[None, :] * stride_k_offset
                + pid_kh * stride_k_h
                + off_d[:, None] * stride_k_d
            )
            k = tl.load(
                k_cache_ptr + k_offsets,
                mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
                other=0.0,
            )

            qk = tl.dot(q, k) * sm_scale_log2e  # [BSQ*H, BLOCKS_PER_STEP*block_size]

            causal = q_token_raw[:, None] >= key_pos[None, :]
            qk = tl.where(causal & pos_mask[None, :], qk, float("-inf"))

            # Reduce + store one score per logical block (BLOCKS_PER_STEP stores).
            for j in tl.static_range(BLOCKS_PER_STEP):
                logical_block_j = logical_base + j
                col_lo = j * block_size
                col_sel = (off_n >= col_lo) & (off_n < col_lo + block_size)
                qk_j = tl.where(col_sel[None, :], qk, float("-inf"))
                sub_max_j = tl.max(qk_j, axis=1)  # [BSQ*H]
                if SCORE_TYPE == "max":
                    score_j = sub_max_j
                else:
                    score_j = sub_max_j + tl.log2(
                        tl.sum(tl.exp2(qk_j - sub_max_j[:, None]), axis=1)
                    )
                    score_j = tl.where(score_j != score_j, float("-inf"), score_j)
                s_offsets_j = (
                    head_flat * stride_s_h
                    + q_token_raw * stride_s_q
                    + logical_block_j * stride_s_n
                )
                tl.store(
                    score_ptr + s_offsets_j,
                    score_j.to(score_ptr.dtype.element_ty),
                    mask=row_valid & (logical_block_j < chunk_end_block),
                )


@triton.jit
def _prefill_bnsd_score_attn_kernel(
    q_ptr,  # [total_q, num_idx_heads, head_dim]  (index Q)
    k_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_dim] (index K)
    v_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_dim] (index V)
    block_table_ptr,  # [all_seqblock_q, max_num_blocks]
    qb_to_qstart_ptr,  # [all_seqblock_q]
    qb_to_qblock_ptr,  # [all_seqblock_q]
    qb_seq_lens_ptr,  # [all_seqblock_q]
    qb_qend_ptr,  # [all_seqblock_q] exclusive q upper bound (cu_seqlens[r+1])
    score_ptr,  # [num_idx_heads, total_q, max_seqblock_k]
    idx_o_ptr,  # [total_q, num_idx_heads, head_dim]
    # scalars
    total_q,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    all_seqblock_q,
    sm_scale,
    # strides
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_offset,
    stride_v_h,
    stride_v_d,
    stride_bt_q,
    stride_bt_n,
    stride_s_h,
    stride_s_q,
    stride_s_n,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    # constexpr meta
    block_size: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
):
    """Fused block-score + index-head dense attention (query-block tiled).

    Each program serially scans ALL its request's KV blocks once, running an online flash-attn softmax that emits ``idx_o`` alongside the block scores.
    """
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_qb = tl.program_id(0)
    pid_kh = tl.program_id(1)

    seq_len = tl.load(qb_seq_lens_ptr + pid_qb).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    q_start = tl.load(qb_to_qstart_ptr + pid_qb).to(tl.int32)
    q_block_local = tl.load(qb_to_qblock_ptr + pid_qb).to(tl.int32)
    q_end = tl.load(qb_qend_ptr + pid_qb).to(tl.int32)

    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_qh = tl.arange(0, BLOCK_SIZE_Q * BLOCK_SIZE_H)
    qi = off_qh // BLOCK_SIZE_H
    hh = off_qh % BLOCK_SIZE_H
    pid_h_base = pid_kh * gqa_group_size
    # q_end (= cu_seqlens[r+1]) bounds to THIS request's real tokens so phantom
    # rows of a partial-tail q-block (block_size_q>1, extend not a BLOCK_SIZE_Q
    # multiple) are masked. Clamp q_token for the Q load to [q_start, q_end-1] so
    # phantom rows never read into the next request's Q data. q_end <= total_q.
    q_token_raw = q_start + q_block_local * BLOCK_SIZE_Q + qi
    row_valid = (q_token_raw < q_end) & (hh < gqa_group_size)
    q_token_flat = tl.maximum(q_start, tl.minimum(q_token_raw, q_end - 1))
    head_flat = pid_h_base + hh

    q_offsets = (
        q_token_flat[:, None] * stride_q_n
        + head_flat[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=row_valid[:, None] & (off_d[None, :] < head_dim),
        other=0.0,
    )

    sm_scale_log2e = sm_scale * 1.4426950409
    # Finite floor (not -inf): an all-masked block has sub_max=-inf -> m_new=m_i,
    # exp2(qk-m_new)=exp2(-inf)=0 (p=0), exp2(m_i-m_new)=1 (acc_o/l_i no-op), so
    # no -inf-​-inf=NaN and no per-iter ``contributes`` where-guard is needed.
    m_i = tl.full((BLOCK_SIZE_Q * BLOCK_SIZE_H,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_Q * BLOCK_SIZE_H,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_Q * BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)

    for logical_block in tl.range(num_blocks):
        physical_block = tl.load(
            block_table_ptr + pid_qb * stride_bt_q + logical_block * stride_bt_n
        ).to(tl.int64)
        key_pos = logical_block * block_size + off_n
        pos_mask = key_pos < seq_len

        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
            other=0.0,
        )
        qk = tl.dot(q, k) * sm_scale_log2e  # [M, N]
        causal = q_token_raw[:, None] >= key_pos[None, :]
        qk = tl.where(causal & pos_mask[None, :], qk, float("-inf"))

        # per-block score (same as the score-only kernel)
        sub_max = tl.max(qk, axis=1)
        if SCORE_TYPE == "max":
            score = sub_max
        else:
            score = sub_max + tl.log2(tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1))
            score = tl.where(score != score, float("-inf"), score)
        s_offsets = (
            head_flat * stride_s_h
            + q_token_raw * stride_s_q
            + logical_block * stride_s_n
        )
        tl.store(
            score_ptr + s_offsets, score.to(score_ptr.dtype.element_ty), mask=row_valid
        )

        # online softmax -> idx_o accumulation
        m_new = tl.maximum(m_i, sub_max)
        # Finite-floor m_i: p=exp2(qk-m_new) is 0 on masked cols (-inf) and on
        # all-masked blocks (m_new=m_i -> no-op), so no contributes guard needed.
        p = tl.exp2(qk - m_new[:, None])
        l_new = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_new)[:, None]
        v_offsets = (
            physical_block * stride_v_block
            + off_n[:, None] * stride_v_offset
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_offsets,
            mask=pos_mask[:, None] & (off_d[None, :] < head_dim),
            other=0.0,
        )
        acc_o = acc_o + tl.dot(p.to(v.dtype), v)
        l_i = l_i * tl.exp2(m_i - m_new) + l_new
        m_i = m_new

    idx_o = acc_o / l_i[:, None]
    o_offsets = (
        q_token_raw[:, None] * stride_o_n
        + head_flat[:, None] * stride_o_h
        + off_d[None, :] * stride_o_d
    )
    tl.store(
        idx_o_ptr + o_offsets,
        idx_o.to(idx_o_ptr.dtype.element_ty),
        mask=row_valid[:, None] & (off_d[None, :] < head_dim),
    )


def _build_qblock_mappings(
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_size_q: int,
    page_size: int,
    max_blocks: int,
    device,
):
    """Precompute per-query-block varlen mappings.

    Returns qb_to_qstart, qb_to_qblock, qb_seq_lens (all [all_seqblock_q] int32), block_table [all_seqblock_q, max_blocks] int32, and all_seqblock_q (int).
    """
    seq_lens_l = seq_lens.to(device=device, dtype=torch.long)
    cu_q = cu_seqlens.to(device=device, dtype=torch.long)
    reqs = req_pool_indices.to(device=device, dtype=torch.long)

    q_lens = cu_q[1:] - cu_q[:-1]  # [bs]
    qb_per_req = (q_lens + block_size_q - 1) // block_size_q  # [bs]
    all_seqblock_q = int(qb_per_req.sum().item())

    # Owning request + q-start (token) per query-block.
    qb_to_req = reqs.repeat_interleave(qb_per_req)  # [all_seqblock_q]
    qb_to_qstart = cu_q[:-1].repeat_interleave(qb_per_req)
    # Per-q-block exclusive upper bound on q_token_flat (= cu_seqlens[r+1] for
    # request r's blocks). Masks the partial tail of a request's last q-block at
    # block_size_q>1: without it, phantom rows (qi past the request's real
    # q_len) have q_token_flat < total_q (== next request's tokens) and would
    # read the next request's Q + score it against THIS request's KV, corrupting
    # the next request's score rows (cross-request contamination).
    qb_qend = cu_q[1:].repeat_interleave(qb_per_req)
    qb_seq_lens = seq_lens_l.repeat_interleave(qb_per_req)
    # Local q-block index within its request.
    cu_blocks = torch.zeros_like(qb_per_req)
    cu_blocks[1:] = qb_per_req[:-1].cumsum(0)
    arange_all = torch.arange(all_seqblock_q, device=device, dtype=torch.long)
    qb_to_qblock = arange_all - cu_blocks.repeat_interleave(qb_per_req)

    # block_table[qb, blk] = physical page of logical block blk of qb's request.
    blk_cols = torch.arange(max_blocks, device=device, dtype=torch.long) * page_size
    max_cols = req_to_token.shape[1]
    blk_cols = blk_cols.clamp(max=max_cols - 1)
    # Advanced-index directly to [all_seqblock_q, max_blocks].
    # DO NOT write `req_to_token[qb_to_req][:, blk_cols]` -- that materializes the
    # intermediate [all_seqblock_q, max_context_len] and OOMs the card. Broadcast
    # the row/column index arrays instead so only the needed [Q, max_blocks] slab
    # is ever allocated.
    token_slots = req_to_token[
        qb_to_req[:, None], blk_cols
    ]  # [all_seqblock_q, max_blocks]
    block_table = (token_slots // page_size).to(torch.int32)

    return (
        qb_to_qstart.to(torch.int32),
        qb_to_qblock.to(torch.int32),
        qb_seq_lens.to(torch.int32),
        qb_qend.to(torch.int32),
        block_table,
        all_seqblock_q,
    )


def flash_prefill_bnsd_score(
    q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    k_cache_bnsd: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    cu_seqlens: torch.Tensor,  # [bs+1]
    seq_lens: torch.Tensor,  # [bs] total KV len
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_size_q: int,
    block_size_k: int,  # == page_size
    sm_scale: float,
    score_type: str = "max",
    num_score_chunks: Optional[int] = None,
    qblock_mappings: Optional[tuple] = None,
) -> torch.Tensor:
    """Block-sparse PREFILL indexer scoring -> score [num_idx_heads, total_q, max_seqblock_k].

    Each query-block (``block_size_q`` tokens) is scored against every KV block of its owning request with per-query-token causal masking, batched in one 2D dot per (query-block, kv-block). ``qblock_mappings`` optionally caches ``_build_qblock_mappings`` output.
    """
    total_q, num_idx_heads, head_dim = q.shape
    num_kv_heads = k_cache_bnsd.shape[2]
    assert num_idx_heads % num_kv_heads == 0
    gqa_group_size = num_idx_heads // num_kv_heads
    page_size = block_size_k
    max_seqblock_k = (int(seq_lens.max().item()) + page_size - 1) // page_size
    max_blocks = max_seqblock_k
    device = q.device

    if qblock_mappings is None:
        (
            qb_to_qstart,
            qb_to_qblock,
            qb_seq_lens,
            qb_qend,
            block_table,
            all_seqblock_q,
        ) = _build_qblock_mappings(
            cu_seqlens,
            seq_lens,
            req_to_token,
            req_pool_indices,
            block_size_q,
            page_size,
            max_blocks,
            device,
        )
    else:
        (
            qb_to_qstart,
            qb_to_qblock,
            qb_seq_lens,
            qb_qend,
            block_table,
            all_seqblock_q,
        ) = qblock_mappings

    if all_seqblock_q == 0:
        return torch.empty(
            (num_idx_heads, total_q, max_seqblock_k), device=device, dtype=torch.float32
        )

    if num_score_chunks is None:
        num_score_chunks = _choose_num_score_chunks(
            max_seqblock_k,
            all_seqblock_q=all_seqblock_q,
            num_kv_heads=num_kv_heads,
        )
    num_score_chunks = max(1, min(num_score_chunks, max_seqblock_k))

    BLOCK_SIZE_Q = _next_power_of_2(block_size_q)

    # Multi-block score tiling (see _prefill_bnsd_score_kernel BLOCKS_PER_STEP).
    # Default 1 == bit-identical to the single-block kernel; set env
    # MINIMAX_NPU_PREFILL_SCORE_BLOCKS_PER_STEP=2 or 4 to fuse that many
    # consecutive KV blocks per tl.dot.
    import os as _os

    _bps_raw = _os.environ.get("MINIMAX_NPU_PREFILL_SCORE_BLOCKS_PER_STEP")
    blocks_per_step = int(_bps_raw) if _bps_raw else 1
    assert blocks_per_step in (1, 2, 4), (
        "MINIMAX_NPU_PREFILL_SCORE_BLOCKS_PER_STEP must be one of 1/2/4, "
        f"got {blocks_per_step!r}"
    )

    score = torch.full(
        (num_idx_heads, total_q, max_seqblock_k),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )

    grid = (all_seqblock_q * num_score_chunks, num_kv_heads)
    _prefill_bnsd_score_kernel[grid](
        q,
        k_cache_bnsd,
        block_table,
        qb_to_qstart,
        qb_to_qblock,
        qb_seq_lens,
        qb_qend,
        score,
        total_q,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        all_seqblock_q,
        num_score_chunks,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache_bnsd.stride(0),
        k_cache_bnsd.stride(1),
        k_cache_bnsd.stride(2),
        k_cache_bnsd.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_size_k,
        BLOCK_SIZE_Q,
        triton.next_power_of_2(gqa_group_size),
        triton.next_power_of_2(head_dim),
        triton.next_power_of_2(page_size * blocks_per_step),
        score_type,
        blocks_per_step,
        num_warps=_SCORE_NW,
        num_stages=_SCORE_NS,
    )
    return score


def flash_prefill_bnsd_score_attn(
    q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]  (index Q)
    k_cache_bnsd: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache_bnsd: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_size_q: int,
    block_size_k: int,
    sm_scale: float,
    score_type: str = "max",
    qblock_mappings: Optional[tuple] = None,
):
    """Fused block-score + index-head dense attention (query-block tiled).

    Returns (score [num_idx_heads, total_q, max_seqblock_k], idx_o [total_q, num_idx_heads, head_dim]): one pass over KV writes per-block scores and accumulates the index heads' online-softmax attention output (idx_o).
    """
    total_q, num_idx_heads, head_dim = q.shape
    num_kv_heads = k_cache_bnsd.shape[2]
    assert num_idx_heads % num_kv_heads == 0
    gqa_group_size = num_idx_heads // num_kv_heads
    page_size = block_size_k
    max_seqblock_k = (int(seq_lens.max().item()) + page_size - 1) // page_size
    max_blocks = max_seqblock_k
    device = q.device

    if qblock_mappings is None:
        (
            qb_to_qstart,
            qb_to_qblock,
            qb_seq_lens,
            qb_qend,
            block_table,
            all_seqblock_q,
        ) = _build_qblock_mappings(
            cu_seqlens,
            seq_lens,
            req_to_token,
            req_pool_indices,
            block_size_q,
            page_size,
            max_blocks,
            device,
        )
    else:
        (
            qb_to_qstart,
            qb_to_qblock,
            qb_seq_lens,
            qb_qend,
            block_table,
            all_seqblock_q,
        ) = qblock_mappings

    BLOCK_SIZE_Q = _next_power_of_2(block_size_q)
    score = torch.full(
        (num_idx_heads, total_q, max_seqblock_k),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    idx_o = torch.zeros(
        (total_q, num_idx_heads, head_dim), device=device, dtype=q.dtype
    )

    if all_seqblock_q > 0:
        grid = (all_seqblock_q, num_kv_heads)
        _prefill_bnsd_score_attn_kernel[grid](
            q,
            k_cache_bnsd,
            v_cache_bnsd,
            block_table,
            qb_to_qstart,
            qb_to_qblock,
            qb_seq_lens,
            qb_qend,
            score,
            idx_o,
            total_q,
            num_kv_heads,
            gqa_group_size,
            head_dim,
            all_seqblock_q,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache_bnsd.stride(0),
            k_cache_bnsd.stride(1),
            k_cache_bnsd.stride(2),
            k_cache_bnsd.stride(3),
            v_cache_bnsd.stride(0),
            v_cache_bnsd.stride(1),
            v_cache_bnsd.stride(2),
            v_cache_bnsd.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            idx_o.stride(0),
            idx_o.stride(1),
            idx_o.stride(2),
            block_size_k,
            BLOCK_SIZE_Q,
            triton.next_power_of_2(gqa_group_size),
            triton.next_power_of_2(head_dim),
            triton.next_power_of_2(page_size),
            score_type,
            num_warps=_SCORE_ATTN_NW,
            num_stages=_SCORE_ATTN_NS,
        )
    return score, idx_o


@triton.jit
def _prefill_topk_from_score_kernel(
    score_ptr,  # [num_kv_heads, total_q, max_seqblock_k] fp32 (-inf for invalid)
    seq_lens_ptr,  # [total_q] int32 -- per-query CAUSAL KV length (prefix+within+1)
    out_ptr,  # [num_kv_heads, total_q, topk + 1] int32 -- topk + appended local block
    total_q,
    max_seqblock_k,
    block_size,
    topk,
    stride_s_h,
    stride_s_q,
    stride_s_k,
    stride_o_h,
    stride_o_q,
    stride_o_t,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TOPK: tl.constexpr,
):
    """Fused block-TopK + causal-local-block append from the score tensor.

    TopK by score selects the causally-valid blocks (score is -inf out-of-range); the local block is appended at slot ``topk`` (-1 if already selected). NOT bitonic -- hits an Ascend broadcast-compare miscompile; argmax ties break to the lowest index, deterministic.
    """
    pid_qb = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_q = tl.arange(0, BLOCK_SIZE_Q)  # [BSQ]
    q_tok = pid_qb * BLOCK_SIZE_Q + off_q  # [BSQ]
    q_valid = q_tok < total_q

    # Per-query causal KV length -> query position + local block. Clamp the load
    # for phantom rows (q_tok >= total_q, partial-tail query-block); their output
    # is masked by q_valid so the value is irrelevant (other=1 -> pos 0).
    seq_len_q = tl.load(seq_lens_ptr + q_tok, mask=q_valid, other=1).to(tl.int32)
    query_pos = tl.maximum(seq_len_q - 1, 0)  # [BSQ]
    # local block = block containing the query's last KV position. block_size is a
    # power of two so // is a shift.
    local_blk = tl.minimum(query_pos // block_size, max_seqblock_k - 1)  # [BSQ]

    off_k = tl.arange(0, BLOCK_SIZE_K)  # [K]
    k_valid = off_k < max_seqblock_k  # [K]

    # Score slab [BSQ, K]; out-of-range / future blocks are -inf (score kernel
    # wrote -inf for key_pos >= seq_len or q_token < key_pos; unwritten blocks
    # past the request stay -inf from the score tensor's -inf init).
    s_offsets = (
        pid_h * stride_s_h + q_tok[:, None] * stride_s_q + off_k[None, :] * stride_s_k
    )
    scores = tl.load(
        score_ptr + s_offsets,
        mask=q_valid[:, None] & k_valid[None, :],
        other=float("-inf"),
    )  # [BSQ, K] fp32

    # TopK selection: TOPK iterations of max + argmax (lowest-index tie-break),
    # each selected block validated + removed before the next. ``tl.range`` (dynamic
    # loop, not unrolled) keeps the loop body small vs ``static_range`` unrolling
    # TOPK copies of the [BSQ, K] max/argmax.
    for rank in tl.range(0, TOPK):
        best = tl.max(scores, axis=1)  # [BSQ]
        is_best = scores == best[:, None]  # [BSQ, K] (== -inf ties handled below)
        pos = tl.min(
            tl.where(is_best, off_k[None, :], BLOCK_SIZE_K), axis=1
        )  # [BSQ] argmax (min index on ties)
        # Store the selected block only if it is genuinely finite-scored
        # (``best > -inf``). Once only -inf blocks remain (future / beyond-request
        # blocks, OR blocks already picked and blanked to -inf below), the rest of
        # the slots are -1 -- WITHOUT this guard a picked-and-blanked block (still
        # a causally-valid index) would be re-selected and pass the range check,
        # duplicating it. ``pos * block_size <= query_pos`` is the same causal
        # validation as ``_append_local_block_to_topk_idx_kernel`` (defensive: in
        # the real score tensor finite <=> causally valid, so this is a subset).
        valid_blk = (
            (best > float("-inf"))
            & (pos < max_seqblock_k)
            & (pos * block_size <= query_pos)
        )
        out_blk = tl.where(valid_blk, pos, -1)
        tl.store(
            out_ptr + pid_h * stride_o_h + q_tok * stride_o_q + rank * stride_o_t,
            tl.where(q_valid, out_blk, -1),
            mask=q_valid,
        )
        # Remove the selected position so the next iteration picks the next best.
        # Blank to -inf (works for finite winners; for -inf tie groups the slot is
        # already -inf so this is a no-op and the guard above emits -1).
        scores = tl.where(off_k[None, :] == pos[:, None], float("-inf"), scores)

    # Append the causal local block at slot TOPK: -1 if it is already among the
    # selected TOPK (dedup, avoids double-counting in the main-attention online
    # softmax), else local_blk. NOTE: dedup via reloading the just-written
    # [BSQ, TOPK] slots is faster than in-loop local_hit accumulation.
    rank_off = tl.arange(0, TOPK)  # [TOPK]
    sel_offsets = (
        pid_h * stride_o_h
        + q_tok[:, None] * stride_o_q
        + rank_off[None, :] * stride_o_t
    )
    selected = tl.load(
        out_ptr + sel_offsets, mask=q_valid[:, None], other=-1
    )  # [BSQ, TOPK]
    local_present = (
        tl.sum((selected == local_blk[:, None]).to(tl.int32), axis=1) > 0
    )  # [BSQ]
    out_local = tl.where(local_present, -1, local_blk)
    tl.store(
        out_ptr + pid_h * stride_o_h + q_tok * stride_o_q + TOPK * stride_o_t,
        tl.where(q_valid, out_local, -1),
        mask=q_valid,
    )


def flash_prefill_bnsd_topk_from_score(
    score: torch.Tensor,  # [num_kv_heads, total_q, max_seqblock_k] fp32
    per_query_seq_lens: torch.Tensor,  # [total_q] int32 -- per-query causal KV len
    topk: int,
    block_size_k: int,
) -> torch.Tensor:
    """Fused TopK + causal-local-block append -> topk_idx [num_kv_heads, total_q, topk+1].

    Local block is already appended at slot ``topk``; the caller's ``_prepare_npu_triton_topk_idx`` must SKIP its own append (detected by ``shape[2] == topk + 1``). Only valid on the index-gqa=1 path (``num_idx_heads == num_kv_heads``).
    """
    num_kv_heads, total_q, max_seqblock_k = score.shape
    assert score.dtype == torch.float32, "score must be fp32"
    assert per_query_seq_lens.dtype == torch.int32
    assert per_query_seq_lens.shape[0] == total_q
    device = score.device

    if total_q == 0:
        return torch.empty(
            (num_kv_heads, 0, topk + 1), dtype=torch.int32, device=device
        )

    out = torch.empty(
        (num_kv_heads, total_q, topk + 1), dtype=torch.int32, device=device
    )

    BLOCK_SIZE_K = triton.next_power_of_2(max_seqblock_k)
    # BSQ is capped at 16 (NOT the UB-safe max of 64). The UB ceiling
    # (bsq*2*K*4 <= 32KB) still binds for large K (e.g. 128K -> BSQ=8).
    bsq = 1
    while bsq < 16 and bsq * 2 * BLOCK_SIZE_K * 4 <= 32768:
        bsq *= 2
    BLOCK_SIZE_Q = max(1, bsq)

    grid = (triton.cdiv(total_q, BLOCK_SIZE_Q), num_kv_heads)
    _prefill_topk_from_score_kernel[grid](
        score,
        per_query_seq_lens,
        out,
        total_q,
        max_seqblock_k,
        block_size_k,
        topk,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        TOPK=topk,
        num_warps=4,
        num_stages=1,
    )
    return out


def flash_prefill_bnsd_indexer(
    q: torch.Tensor,  # idx_q [total_q, num_idx_heads, idx_dim]
    k_cache_bnsd: torch.Tensor,  # idx_k
    v_cache_bnsd: torch.Tensor,  # idx_v
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    sm_scale: float,
    score_type: str = "max",
    qblock_mappings: Optional[tuple] = None,
    per_query_seq_lens: Optional[torch.Tensor] = None,
):
    """Prefill indexer (fused): returns (idx_o, topk_idx [num_idx_heads, total_q, topk]).

    topk_idx is UN-reduced (per index head); caller does GQA reduction. With ``per_query_seq_lens`` and num_idx_heads == num_kv_heads, returns ``[num_kv_heads, total_q, topk + 1]`` (local appended, caller skips append); else ``[..., topk]``.
    """
    score, idx_o = flash_prefill_bnsd_score_attn(
        q,
        k_cache_bnsd,
        v_cache_bnsd,
        cu_seqlens,
        seq_lens,
        req_to_token,
        req_pool_indices,
        block_size_q,
        block_size_k,
        sm_scale,
        score_type,
        qblock_mappings,
    )
    if per_query_seq_lens is not None and q.shape[1] == k_cache_bnsd.shape[2]:
        # Fused TopK + local-block append -> [num_kv_heads, total_q, topk + 1].
        return idx_o, flash_prefill_bnsd_topk_from_score(
            score, per_query_seq_lens, topk, block_size_k
        )
    max_seqblock_k = score.shape[-1]
    actual_topk = min(topk, max_seqblock_k)
    _, idx = torch.topk(score, k=actual_topk, dim=-1)  # [num_idx_heads, total_q, k]
    idx = idx.to(torch.int32)
    if actual_topk < topk:
        pad = torch.full(
            (idx.shape[0], idx.shape[1], topk - actual_topk),
            -1,
            device=idx.device,
            dtype=idx.dtype,
        )
        idx = torch.cat([idx, pad], dim=-1)
    return idx_o, idx.contiguous()


def flash_prefill_bnsd_with_topk_idx(
    q: torch.Tensor,  # idx_q [total_q, num_idx_heads, idx_dim]
    k_cache_bnsd: torch.Tensor,  # idx_k [num_pages, page_size, num_kv_heads, idx_dim]
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    sm_scale: float,
    score_type: str = "max",
    qblock_mappings: Optional[tuple] = None,
    per_query_seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Prefill indexer: score (varlen, batched over query-blocks) + topk.

    Returns topk_idx [num_idx_heads, total_q, topk] UN-reduced (per index head); caller does GQA reduction. Score-only (no idx_o) for ``disable_index_value``. With ``per_query_seq_lens`` and num_idx_heads == num_kv_heads, returns ``[..., topk + 1]`` (local appended, caller skips append).
    """

    max_seqblock_k = (int(seq_lens.max().item()) + block_size_k - 1) // block_size_k
    score = flash_prefill_bnsd_score(
        q,
        k_cache_bnsd,
        cu_seqlens,
        seq_lens,
        req_to_token,
        req_pool_indices,
        block_size_q,
        block_size_k,
        sm_scale,
        score_type,
        qblock_mappings=qblock_mappings,
    )
    if per_query_seq_lens is not None and q.shape[1] == k_cache_bnsd.shape[2]:
        return flash_prefill_bnsd_topk_from_score(
            score, per_query_seq_lens, topk, block_size_k
        )
    actual_topk = min(topk, max_seqblock_k)
    idx = torch.topk(score, k=actual_topk, dim=-1).indices.to(torch.int32)
    if actual_topk < topk:
        pad = torch.full(
            (idx.shape[0], idx.shape[1], topk - actual_topk),
            -1,
            device=idx.device,
            dtype=idx.dtype,
        )
        idx = torch.cat([idx, pad], dim=-1)
    return idx.contiguous()
