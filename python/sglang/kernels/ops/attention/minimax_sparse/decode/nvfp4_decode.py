"""Decode-side sparse attention over the packed NVFP4 (MSA-v1) KV cache.

The verify scorer mirrors its fp8 counterpart (`verify_topk_idx`) but reads a cache
row as its even and odd columns (see `nvfp4_load_even_odd`), so the dots run on
``[tokens, D//2]`` tiles against the matching halves of Q and no interleave or
transpose of the cache tile is needed. The attention itself runs on gfx950's Gluon
kernel (`nvfp4_decode_gfx950`).
"""

from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_gfx95_supported

from ..common.nvfp4 import Nvfp4Cache, nvfp4_load_even_odd, nvfp4_slot_offsets
from ..common.utils import unit_scale
from .flash_with_topk_idx import decode_topk_uses_jit, select_decode_topk


@triton.jit
def _verify_score_nvfp4_kernel(
    q_ptr,  # Q: rows x qh x d, rows = num_reqs * NUM_DRAFT
    k_payload_ptr,
    k_scale_ptr,
    req_to_token_ptr,  # max_reqs x max_kv_len
    score_ptr,  # Score: qh x rows x max_seqblock
    seq_lens,  # [rows]
    slot_ids,  # [rows]
    max_slots,
    num_reqs,
    gqa_group_size,
    topk,
    sm_scale,
    init_blocks,
    local_blocks,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    pay_page,
    pay_head,
    pay_row,
    sc_page,
    sc_head,
    sc_row,
    stride_r2t_b,
    stride_s_h,
    stride_s_b,
    stride_s_n,
    block_size: tl.constexpr,
    NUM_DRAFT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # >= NUM_DRAFT * gqa_group_size
    HEAD_DIM: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    SKIP_TRIVIAL_TOPK_SCORE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_rc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_r = pid_rc % num_reqs
    pid_c = pid_rc // num_reqs
    # M column m holds draft row m // gqa_group_size, head m % gqa_group_size.
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
    # All draft rows of a request share one req_to_token row.
    sid = (tl.load(slot_ids + pid_r * NUM_DRAFT).to(tl.int64) + max_slots) % max_slots
    r2t_base = req_to_token_ptr + sid * stride_r2t_b
    # Q^T halves [HEAD_DIM // 2, M]; fp8 and bf16 Q widen to fp16 exactly in range.
    half = tl.arange(0, HEAD_DIM // 2)
    q_base = q_ptr + m_row[None, :] * stride_q_b + m_head[None, :] * stride_q_h
    qt_even = tl.load(
        q_base + (2 * half)[:, None] * stride_q_d, mask=m_valid[None, :], other=0.0
    ).to(tl.float16)
    qt_odd = tl.load(
        q_base + (2 * half + 1)[:, None] * stride_q_d, mask=m_valid[None, :], other=0.0
    ).to(tl.float16)
    m_local_start = tl.maximum(0, m_num_blocks - local_blocks)
    s_row_ptr = score_ptr + m_head * stride_s_h + m_row * stride_s_b
    off_n = tl.arange(0, block_size)
    for blk in range(chunk_start_block, chunk_end_block):
        pos = blk * block_size + off_n
        pos_mask = pos < seq_len
        base_slot = tl.load(r2t_base + blk * block_size).to(tl.int64)
        slots = (base_slot + max_slots) % max_slots + off_n
        k_even, k_odd = nvfp4_load_even_odd(
            k_payload_ptr,
            k_scale_ptr,
            nvfp4_slot_offsets(slots, pay_page, pay_head, pay_row, pid_kh, PAGE_SIZE),
            nvfp4_slot_offsets(slots, sc_page, sc_head, sc_row, pid_kh, PAGE_SIZE),
            pos_mask,
            HEAD_DIM,
        )
        s = tl.dot(k_even, qt_even) + tl.dot(k_odd, qt_odd)  # [block, M]
        s = s * sm_scale_log2e + tl.where(
            pos[:, None] < m_seq_len[None, :], 0, float("-inf")
        )
        score = tl.max(s, axis=0)
        is_init = blk < init_blocks
        is_local = (blk >= m_local_start) & (blk < m_num_blocks)
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            s_row_ptr + blk * stride_s_n,
            score.to(score_ptr.dtype.element_ty),
            mask=m_valid & (blk < m_num_blocks),
        )


def verify_topk_idx_nvfp4(
    *,
    q: torch.Tensor,
    k_cache: Nvfp4Cache,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen: int,
    slot_ids: torch.Tensor,
    num_draft: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float],
    q_scale: Optional[float],
    k_scale: Optional[float],
    topk_idx_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`verify_topk_idx` over an NVFP4 index-K cache."""
    rows, num_q_heads, head_dim = q.shape
    page_size = k_cache.page_size
    assert head_dim == 128 and page_size % block_size == 0
    max_slots = k_cache.payload.shape[0] * page_size
    num_kv_heads = k_cache.num_heads
    num_reqs = rows // num_draft
    assert num_reqs * num_draft == rows
    gqa_group_size = num_q_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    sm_scale = sm_scale * unit_scale(q_scale) * unit_scale(k_scale)
    score_kv_len = min(max_seqlen, req_to_token.shape[1])
    # no -inf fill: columns past a row's ceil(seq_len/block) are never read
    score = torch.empty(
        (num_q_heads, rows, triton.cdiv(score_kv_len, block_size)),
        dtype=torch.float32,
        device=q.device,
    )
    use_jit_topk = decode_topk_uses_jit(score.shape[2], topk)
    # The fp8 kernel's split-K sizing, but one or two requests fill 1024 programs (MI350X)
    groups = max(1, num_reqs * num_kv_heads)
    target = max(1, min(256, 2048 // groups), 1024 // groups)
    num_kv_chunks = 1 << (target.bit_length() - 1)
    block_m = max(16, triton.next_power_of_2(num_draft * gqa_group_size))
    _verify_score_nvfp4_kernel[(num_reqs * num_kv_chunks, num_kv_heads)](
        q,
        k_cache.payload,
        k_cache.scales,
        req_to_token,
        score,
        seq_lens,
        slot_ids,
        max_slots,
        num_reqs,
        gqa_group_size,
        topk,
        sm_scale,
        init_blocks,
        local_blocks,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        *k_cache.payload_strides(),
        *k_cache.scale_strides(),
        req_to_token.stride(0),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_size=block_size,
        NUM_DRAFT=num_draft,
        BLOCK_SIZE_M=block_m,
        HEAD_DIM=head_dim,
        NUM_KV_CHUNKS=num_kv_chunks,
        SKIP_TRIVIAL_TOPK_SCORE=use_jit_topk,
        PAGE_SIZE=page_size,
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


def _use_nvfp4_decode_gfx950(
    q: torch.Tensor,
    k_cache: Nvfp4Cache,
    v_cache: Nvfp4Cache,
    topk_idx: torch.Tensor,
    block_size: int,
) -> bool:
    if not is_gfx95_supported():
        return False
    try:
        from .nvfp4_decode_gfx950 import can_use_nvfp4_decode_gfx950
    except ImportError:  # Gluon without the CDNA4 MFMA op
        return False
    return can_use_nvfp4_decode_gfx950(q, k_cache, v_cache, topk_idx, block_size)


@torch.no_grad()
def flash_decode_with_gqa_share_sparse_nvfp4(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    k_cache: Nvfp4Cache,
    v_cache: Nvfp4Cache,
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    seq_lens: torch.Tensor,  # [batch_size, ]
    slot_ids: torch.Tensor,  # [batch_size, ]
    block_size: int,
    topk_idx: torch.Tensor,  # [num_kv_heads, batch_size, topk]
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    out_quant: str = "",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """`flash_decode_with_gqa_share_sparse` over NVFP4 main K/V (no sink, no hisparse),
    on gfx950's Gluon kernel. ``out_quant``: see ``flash_decode_nvfp4_gfx950``."""
    if not _use_nvfp4_decode_gfx950(q, k_cache, v_cache, topk_idx, block_size):
        raise NotImplementedError(
            "NVFP4 sparse decode runs on the gfx950 Gluon kernel only, which needs block "
            "size 128, 16 query heads per KV head and each layer's NVFP4 cache under 2 "
            "GiB; lower --mem-fraction-static for a smaller cache"
        )
    from .nvfp4_decode_gfx950 import flash_decode_nvfp4_gfx950

    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return flash_decode_nvfp4_gfx950(
        q,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens,
        slot_ids,
        block_size,
        topk_idx,
        sm_scale=sm_scale * unit_scale(q_scale) * unit_scale(k_scale),
        v_scale=unit_scale(v_scale),
        out_quant=out_quant,
    )
