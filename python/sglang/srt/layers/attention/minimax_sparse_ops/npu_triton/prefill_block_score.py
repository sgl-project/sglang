"""NPU varlen multi-token sparse-attention PREFILL kernels (Approach A, Phase A1).

Replaces the per-query-flattening reuse of the decode kernels
(``_forward_npu_triton_prefill`` via ``flash_decode_bnsd_with_topk_idx``), which
is correct but ~2x slower than the PyTorch masked-full path: the decode kernels
are built for few-queries / long-context and collapse their split-K tiling when
fed a 512-token prefill chunk as 512 per-query rows.

These kernels instead batch the block-scoring across a *query block* of
``BLOCK_SIZE_Q`` tokens -- mirroring the GPU ``prefill/`` varlen kernels
(``flash_prefill_with_topk_index``) -- so the indexer pays no per-query launch
overhead and the seq dimension stays parallelized.

Faithfully extends the validated NPU decode score kernel
(``_decode_bnsd_score_chunk_kernel`` in flash_block_score_decode.py):
  * Q grows from ``[H, D]`` (one query) to ``[BLOCK_SIZE_Q * H, D]`` (a query
    block); the QK dot stays 2D -- NO 3D reshape of the dot result, which
    miscompiles / ~1500x-slows on this Ascend TBE backend (WARNING at
    flash_block_score_decode.py:525-532).
  * varlen: the host precomputes per-query-block mappings (owning request,
    q-start, seq_len, prefix_len, block_table) so the kernel gathers by
    ``pid_qb`` -- no in-kernel varlen reverse search.
  * Ascend-safe conventions: power-of-2 block sizes, ``tl.range`` for variable
    loops, masked stores.

STATUS (Phase A1): score kernel + ``torch.topk`` topk path. Needs Ascend
compile-test + correctness diff vs the PyTorch ``_select_sparse_blocks`` path.
Streaming-Triton topk (Phase A2) and the main sparse kernel (Phase A3) follow.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.flash_block_score_decode import (
    _choose_num_score_chunks,
    _next_power_of_2,
)


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
):
    """Score one query-block x one kv_head tile.

    Grid: ``(all_seqblock_q * num_score_chunks, num_kv_heads)``. ``pid_qbc %
    all_seqblock_q`` -> query-block; ``// all_seqblock_q`` -> score chunk (tile
    of consecutive KV blocks, so program count is independent of context length
    -- same trick as ``_decode_bnsd_score_chunk_kernel``).
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
    # phantom rows never read across the request boundary into the next request's
    # Q data, wasting memory bandwidth. q_end <= total_q always.
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
            score = sub_max + tl.log2(tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1))
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

    Same Q/K load and per-block score write as ``_prefill_bnsd_score_kernel``,
    but each program loops over ALL of its request's KV blocks once and also runs
    an online (flash-attn) softmax -> produces ``idx_o`` (the index heads' dense
    causal attention output) alongside the block scores. No chunking / merge:
    grid is ``(all_seqblock_q, num_kv_heads)`` and each program serially scans
    its blocks. Parallelism comes from (query-block x kv_head); fine because
    index heads are few and query-block count scales with batch x chunk.
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
    # phantom rows never read across the request boundary into the next request's
    # Q data, wasting memory bandwidth. q_end <= total_q always.
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
    m_i = tl.full((BLOCK_SIZE_Q * BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
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
            head_flat * stride_s_h + q_token_raw * stride_s_q + logical_block * stride_s_n
        )
        tl.store(
            score_ptr + s_offsets, score.to(score_ptr.dtype.element_ty), mask=row_valid
        )

        # online softmax -> idx_o accumulation
        m_new = tl.maximum(m_i, sub_max)
        # guard exp2(-inf - -inf)=nan on fully-masked rows: only update where
        # the block contributes (sub_max > -inf).
        contributes = sub_max > float("-inf")
        p = tl.where(contributes[:, None], tl.exp2(qk - m_new[:, None]), 0.0)
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
        acc_o = tl.where(contributes[:, None], acc_o + tl.dot(p.to(v.dtype), v), acc_o)
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
    """Precompute per-query-block varlen mappings (cheap PyTorch).

    Returns qb_to_qstart, qb_to_qblock, qb_seq_lens (all [all_seqblock_q] int32),
    block_table [all_seqblock_q, max_blocks] int32, and all_seqblock_q (int).
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
    # Advanced-index directly to [all_seqblock_q, max_blocks] (12 MB at 128K).
    # DO NOT write `req_to_token[qb_to_req][:, blk_cols]` -- that materializes the
    # intermediate [all_seqblock_q, max_context_len] (= 1.6 GB at ctx 131072) and
    # OOMs the card (only ~600 MB free after weights+KV+capture). Broadcast the
    # row/column index arrays instead so only the needed [Q, max_blocks] slab is
    # ever allocated.
    token_slots = req_to_token[qb_to_req[:, None], blk_cols]  # [all_seqblock_q, max_blocks]
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
) -> torch.Tensor:
    """Block-sparse PREFILL indexer scoring -> score [num_idx_heads, total_q, max_seqblock_k].

    Each query-block (``block_size_q`` tokens) is scored against every KV block
    of its owning request with per-query-token causal masking, batched in one
    2D dot per (query-block, kv-block). varlen multi-token analogue of the decode
    score kernel; the dominant prefill cost.
    """
    total_q, num_idx_heads, head_dim = q.shape
    num_kv_heads = k_cache_bnsd.shape[2]
    assert num_idx_heads % num_kv_heads == 0
    gqa_group_size = num_idx_heads // num_kv_heads
    page_size = block_size_k
    max_seqblock_k = (int(seq_lens.max().item()) + page_size - 1) // page_size
    max_blocks = max_seqblock_k
    device = q.device

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
        triton.next_power_of_2(page_size),
        score_type,
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
):
    """Fused block-score + index-head dense attention (query-block tiled).

    Returns (score [num_idx_heads, total_q, max_seqblock_k], idx_o
    [total_q, num_idx_heads, head_dim]). One pass over KV: writes per-block
    scores (for topk) AND accumulates the index heads' online-softmax attention
    output (idx_o) -- replaces a separate index dense-attention pass.
    """
    total_q, num_idx_heads, head_dim = q.shape
    num_kv_heads = k_cache_bnsd.shape[2]
    assert num_idx_heads % num_kv_heads == 0
    gqa_group_size = num_idx_heads // num_kv_heads
    page_size = block_size_k
    max_seqblock_k = (int(seq_lens.max().item()) + page_size - 1) // page_size
    max_blocks = max_seqblock_k
    device = q.device

    (
        qb_to_qstart,
        qb_to_qblock,
        qb_seq_lens,
        qb_qend,
        block_table,
        all_seqblock_q,
    ) = _build_qblock_mappings(
        cu_seqlens, seq_lens, req_to_token, req_pool_indices,
        block_size_q, page_size, max_blocks, device,
    )

    BLOCK_SIZE_Q = _next_power_of_2(block_size_q)
    score = torch.full(
        (num_idx_heads, total_q, max_seqblock_k), float("-inf"),
        device=device, dtype=torch.float32,
    )
    idx_o = torch.zeros((total_q, num_idx_heads, head_dim), device=device, dtype=q.dtype)

    if all_seqblock_q > 0:
        grid = (all_seqblock_q, num_kv_heads)
        _prefill_bnsd_score_attn_kernel[grid](
            q, k_cache_bnsd, v_cache_bnsd, block_table,
            qb_to_qstart, qb_to_qblock, qb_seq_lens, qb_qend, score, idx_o,
            total_q, num_kv_heads, gqa_group_size, head_dim, all_seqblock_q, sm_scale,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache_bnsd.stride(0), k_cache_bnsd.stride(1), k_cache_bnsd.stride(2), k_cache_bnsd.stride(3),
            v_cache_bnsd.stride(0), v_cache_bnsd.stride(1), v_cache_bnsd.stride(2), v_cache_bnsd.stride(3),
            block_table.stride(0), block_table.stride(1),
            score.stride(0), score.stride(1), score.stride(2),
            idx_o.stride(0), idx_o.stride(1), idx_o.stride(2),
            block_size_k,
            BLOCK_SIZE_Q,
            triton.next_power_of_2(gqa_group_size),
            triton.next_power_of_2(head_dim),
            triton.next_power_of_2(page_size),
            score_type,
        )
    return score, idx_o


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
):
    """Prefill indexer (fused): returns (idx_o, topk_idx [num_idx_heads, total_q, topk]).

    topk_idx is UN-reduced (per index head) to match the decode
    ``flash_decode_bnsd_with_topk_idx`` contract; the caller does GQA reduction
    via ``topk_index_reduce``.
    """
    score, idx_o = flash_prefill_bnsd_score_attn(
        q, k_cache_bnsd, v_cache_bnsd, cu_seqlens, seq_lens,
        req_to_token, req_pool_indices, block_size_q, block_size_k,
        sm_scale, score_type,
    )
    max_seqblock_k = score.shape[-1]
    actual_topk = min(topk, max_seqblock_k)
    _, idx = torch.topk(score, k=actual_topk, dim=-1)  # [num_idx_heads, total_q, k]
    idx = idx.to(torch.int32)
    if actual_topk < topk:
        pad = torch.full(
            (idx.shape[0], idx.shape[1], topk - actual_topk),
            -1, device=idx.device, dtype=idx.dtype,
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
) -> torch.Tensor:
    """Prefill indexer: score (varlen, batched over query-blocks) + topk.

    Returns topk_idx [num_idx_heads, total_q, topk] UN-reduced (per index head);
    caller does GQA reduction. Score-only variant (no idx_o) for the
    ``disable_index_value`` case.
    """
    score = flash_prefill_bnsd_score(
        q, k_cache_bnsd, cu_seqlens, seq_lens,
        req_to_token, req_pool_indices, block_size_q, block_size_k,
        sm_scale, score_type,
    )
    max_seqblock_k = score.shape[-1]
    actual_topk = min(topk, max_seqblock_k)
    idx = torch.topk(score, k=actual_topk, dim=-1).indices.to(torch.int32)
    if actual_topk < topk:
        pad = torch.full(
            (idx.shape[0], idx.shape[1], topk - actual_topk),
            -1, device=idx.device, dtype=idx.dtype,
        )
        idx = torch.cat([idx, pad], dim=-1)
    return idx.contiguous()
