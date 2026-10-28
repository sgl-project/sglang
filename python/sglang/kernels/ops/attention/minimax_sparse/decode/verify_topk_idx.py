"""Linear-chain target-verify indexer for MiniMax sparse attention.

A linear draft chain verifies ``num_draft`` consecutive queries per request. They
share one KV prefix and differ only by causal length, so the score kernel places a
request's draft rows in the MMA M dimension and reads each index-K tile once
instead of once per row. Each row keeps its own causal and local-block masks, so
its scores and top-k match the per-row decode indexer.
"""

from typing import Optional, Union

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_gfx95_supported

from ..common.nvfp4 import Nvfp4Cache
from ..common.utils import check_sparse_kv_fp8, unit_scale
from .flash_with_topk_idx import decode_topk_uses_jit, select_decode_topk
from .nvfp4_decode import verify_topk_idx_nvfp4


@triton.jit
def _verify_score_kernel(
    q_ptr,  # Q: rows x qh x d, rows = num_reqs * NUM_DRAFT
    k_cache_ptr,  # K paged: max_slots x kh x d
    req_to_token_ptr,  # max_reqs x max_kv_len
    score_ptr,  # Score: qh x rows x max_seqblock
    seq_lens,  # [rows]
    slot_ids,  # [rows]
    max_slots,
    num_reqs,
    gqa_group_size,
    head_dim,
    topk,
    sm_scale,
    k_scale,
    init_blocks,
    local_blocks,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_r2t_b,
    stride_s_h,
    stride_s_b,
    stride_s_n,
    block_size: tl.constexpr,
    NUM_DRAFT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # >= NUM_DRAFT * gqa_group_size
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    SKIP_TRIVIAL_TOPK_SCORE: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    BLOCKS_PER_K_BLOCK: tl.constexpr = BLOCK_SIZE_N // block_size
    pid_rc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_r = pid_rc % num_reqs
    pid_c = pid_rc // num_reqs
    # M row m holds draft row m // gqa_group_size, head m % gqa_group_size.
    off_m = tl.arange(0, BLOCK_SIZE_M)
    m_draft = off_m // gqa_group_size
    m_head = pid_kh * gqa_group_size + off_m % gqa_group_size
    m_valid = m_draft < NUM_DRAFT
    m_row = pid_r * NUM_DRAFT + tl.minimum(m_draft, NUM_DRAFT - 1)
    m_seq_len = tl.load(seq_lens + m_row).to(tl.int32)
    m_num_blocks = (m_seq_len + block_size - 1) // block_size
    num_blocks = tl.max(tl.where(m_valid, m_num_blocks, 0), axis=0)
    if SKIP_TRIVIAL_TOPK_SCORE:
        if num_blocks <= topk:
            return
    chunk_size_blocks = tl.cdiv(num_blocks, NUM_KV_CHUNKS)
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return
    seq_len = tl.max(tl.where(m_valid, m_seq_len, 0), axis=0)
    chunk_start = chunk_start_block * block_size
    chunk_end = tl.minimum(chunk_end_block * block_size, seq_len)
    # All draft rows of a request share one req_to_token row.
    sid = (tl.load(slot_ids + pid_r * NUM_DRAFT).to(tl.int64) + max_slots) % max_slots
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_bpk = tl.arange(0, BLOCKS_PER_K_BLOCK)
    dim_mask = off_d < head_dim
    q = tl.load(
        q_ptr
        + m_row[:, None] * stride_q_b
        + m_head[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=m_valid[:, None] & dim_mask[None, :],
        other=0.0,
    )
    m_local_start = tl.maximum(0, m_num_blocks - local_blocks)
    r2t_base = req_to_token_ptr + sid * stride_r2t_b
    s_row_ptr = score_ptr + m_head * stride_s_h + m_row * stride_s_b
    for i in range(chunk_start, chunk_end, BLOCK_SIZE_N):
        pos = i + off_n
        pos_mask = pos < chunk_end
        slots = tl.load(r2t_base + pos, mask=pos_mask, other=0).to(tl.int64)
        slots = (slots + max_slots) % max_slots
        k = tl.load(
            k_cache_ptr
            + slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=dim_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if IS_FP8:
            k = k.to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        qk += tl.where(pos[None, :] < m_seq_len[:, None], 0, float("-inf"))
        qk += tl.dot(q, k) * (sm_scale_log2e * k_scale)
        score = tl.max(
            tl.reshape(
                qk, (BLOCK_SIZE_M, BLOCKS_PER_K_BLOCK, block_size), can_reorder=False
            ),
            axis=2,
        )
        blk = i // block_size + off_bpk
        is_init = blk[None, :] < init_blocks
        is_local = (blk[None, :] >= m_local_start[:, None]) & (
            blk[None, :] < m_num_blocks[:, None]
        )
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            s_row_ptr[:, None] + blk[None, :] * stride_s_n,
            score.to(score_ptr.dtype.element_ty),
            mask=m_valid[:, None]
            & (blk[None, :] < tl.minimum(m_num_blocks, chunk_end_block)[:, None]),
        )


def _use_fused_verify_topk(
    q: torch.Tensor,
    k_cache: Union[torch.Tensor, Nvfp4Cache],
    *,
    num_draft: int,
    topk: int,
    local_blocks: int,
    page_size: int,
    block_size: int,
) -> bool:
    if not is_gfx95_supported():
        return False
    try:
        from .verify_topk_gfx950 import can_fuse_verify_topk
    except ImportError:  # Gluon without the CDNA4 MFMA op
        return False
    return can_fuse_verify_topk(
        q, k_cache, num_draft, topk, local_blocks, page_size, block_size
    )


@torch.no_grad()
def verify_topk_idx(
    *,
    q: torch.Tensor,  # [rows, num_heads, head_dim], rows = num_reqs * num_draft
    k_cache: Union[
        torch.Tensor, Nvfp4Cache
    ],  # [max_slots, num_kv_heads, head_dim] or NVFP4
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    seq_lens: torch.Tensor,  # [rows]
    max_seqlen: int,
    slot_ids: torch.Tensor,  # [rows]
    num_draft: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float],
    q_scale: Optional[float],
    k_scale: Optional[float],
    topk_idx_out: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    """Top-k block ids per (head, row), as flash_decode_with_topk_idx returns them."""
    is_nvfp4 = isinstance(k_cache, Nvfp4Cache)
    if _use_fused_verify_topk(
        q,
        k_cache,
        num_draft=num_draft,
        topk=topk,
        local_blocks=local_blocks,
        page_size=page_size,
        block_size=block_size,
    ):
        from .verify_topk_gfx950 import fused_verify_topk_idx

        sm_scale = (q.shape[-1] ** -0.5 if sm_scale is None else sm_scale) * unit_scale(
            q_scale
        )
        # Each score kernel's fp32 scale order; the NVFP4 one folds k_scale in first.
        if is_nvfp4:
            sm_scale, k_scale = sm_scale * unit_scale(k_scale), None
        return fused_verify_topk_idx(
            q,
            k_cache,
            req_to_token,
            seq_lens,
            max_seqlen,
            slot_ids,
            num_draft,
            block_size,
            topk,
            init_blocks,
            local_blocks,
            sm_scale=sm_scale,
            k_scale=unit_scale(k_scale),
            out=topk_idx_out,
        )
    if is_nvfp4:
        return verify_topk_idx_nvfp4(
            q=q,
            k_cache=k_cache,
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            max_seqlen=max_seqlen,
            slot_ids=slot_ids,
            num_draft=num_draft,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            sm_scale=sm_scale,
            q_scale=q_scale,
            k_scale=k_scale,
            topk_idx_out=topk_idx_out,
        )
    is_fp8 = check_sparse_kv_fp8(q, k_cache, None, label="verify indexer")
    rows, num_q_heads, head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    num_reqs = rows // num_draft
    assert num_reqs * num_draft == rows
    gqa_group_size = num_q_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    sm_scale = sm_scale * unit_scale(q_scale)
    score_kv_len = min(max_seqlen, req_to_token.shape[1])
    # no -inf fill: columns past a row's ceil(seq_len/block) are never read
    score = torch.empty(
        (num_q_heads, rows, triton.cdiv(score_kv_len, block_size)),
        dtype=torch.float32,
        device=q.device,
    )
    use_jit_topk = decode_topk_uses_jit(score.shape[2], topk)
    # measured on gfx950 at 1/8/32 requests, 8k context
    TARGET_GRID = 2048
    target = max(1, min(256, TARGET_GRID // max(1, num_reqs * num_kv_heads)))
    num_kv_chunks = 1 << (target.bit_length() - 1)
    block_m = max(16, triton.next_power_of_2(num_draft * gqa_group_size))
    _verify_score_kernel[(num_reqs * num_kv_chunks, num_kv_heads)](
        q,
        k_cache,
        req_to_token,
        score,
        seq_lens,
        slot_ids,
        max_slots,
        num_reqs,
        gqa_group_size,
        head_dim,
        topk,
        sm_scale,
        unit_scale(k_scale),
        init_blocks,
        local_blocks,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        req_to_token.stride(0),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_size=block_size,
        NUM_DRAFT=num_draft,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_size,
        BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
        NUM_KV_CHUNKS=num_kv_chunks,
        SKIP_TRIVIAL_TOPK_SCORE=use_jit_topk,
        IS_FP8=is_fp8,
        num_warps=4,
        num_stages=1,
    )
    return select_decode_topk(
        score=score,
        seq_lens=seq_lens,
        block_size=block_size,
        topk=topk,
        use_jit_topk=use_jit_topk,
        out=topk_idx_out,
    )
