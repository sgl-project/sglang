"""SM120 multi-head tiled FlashMLA sparse decode kernel.

Key optimization over the upstream `_tiled_sparse_decode_kernel`:
- Grid is (B, ceil(H / H_TILE)) instead of (B, H). Each program processes
  H_TILE heads of the same batch element.
- The paged KV gather (the dominant memory traffic) is performed once per
  tile and reused across H_TILE heads, slashing global-memory traffic by
  H_TILE.
- QK scores and acc accumulation use `tl.dot` so Blackwell's tensor cores
  see the BF16 mma.

Layout assumptions match the upstream kernel:
  Page layout per token (576 data bytes + 8 scale bytes):
    [0:448]   FP8(e4m3fn) nope values   (448 elements)
    [448:576] BF16        rope values   (64 elements = 128 bytes)
    [page_size*576 + offs*8 : +7]  UE8M0 scale bytes (7 groups of 64)
"""

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

LOG2E = tl.constexpr(1.4426950408889634)

_NOPE_DIM = 448
_ROPE_DIM = 64
_TOKEN_DATA_STRIDE = 576


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 16, "H_TILE": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 32, "H_TILE": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 16, "H_TILE": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 32, "H_TILE": 8}, num_warps=8, num_stages=2),
    ],
    key=["topk_rounded", "H"],
)
@triton.jit
def _mhead_tiled_sparse_decode_kernel(
    Q_ptr,
    cache_fp8_ptr,
    cache_uint8_ptr,
    cache_bf16_ptr,
    indices_ptr,
    topk_len_ptr,
    O_ptr,
    LSE_ptr,
    sm_scale: tl.float32,
    page_size: tl.int32,
    page_bytes: tl.int64,
    scale_section_off: tl.int64,
    H: tl.int32,
    topk: tl.int32,
    topk_rounded: tl.int32,
    has_topk_len: tl.constexpr,
    stride_qb: tl.int32,
    stride_qh: tl.int32,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    stride_ib: tl.int32,
    NOPE_PAD: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM_RT: tl.int32,
    BLOCK_T: tl.constexpr,
    H_TILE: tl.constexpr,
):
    bid = tl.program_id(0)
    h_block = tl.program_id(1)

    h_offs = h_block * H_TILE + tl.arange(0, H_TILE)  # [H_TILE]
    h_mask = h_offs < H

    nope_offs = tl.arange(0, NOPE_PAD)  # [NOPE_PAD]
    nope_mask = nope_offs < NOPE_DIM_RT
    rope_offs = tl.arange(0, ROPE_DIM)

    # ---- Load Q for H_TILE heads: [H_TILE, NOPE_PAD] and [H_TILE, ROPE_DIM] ----
    q_base_b = bid * stride_qb
    q_nope_addr = q_base_b + h_offs[:, None] * stride_qh + nope_offs[None, :]
    q_nope_mask2 = h_mask[:, None] & nope_mask[None, :]
    q_nope = tl.load(Q_ptr + q_nope_addr, mask=q_nope_mask2, other=0.0)
    q_nope = q_nope.to(tl.float32) * sm_scale

    q_rope_addr = (
        q_base_b + h_offs[:, None] * stride_qh + NOPE_DIM_RT + rope_offs[None, :]
    )
    q_rope = tl.load(Q_ptr + q_rope_addr, mask=h_mask[:, None], other=0.0)
    q_rope = q_rope.to(tl.float32) * sm_scale

    # ---- Per-head online softmax state ----
    valid_topk = topk
    if has_topk_len:
        valid_topk = tl.load(topk_len_ptr + bid).to(tl.int32)
        valid_topk = tl.minimum(valid_topk, topk)

    m_i = tl.zeros([H_TILE], dtype=tl.float32) - 1e30
    l_i = tl.zeros([H_TILE], dtype=tl.float32)
    acc_nope = tl.zeros([H_TILE, NOPE_PAD], dtype=tl.float32)
    acc_rope = tl.zeros([H_TILE, ROPE_DIM], dtype=tl.float32)

    group_ids = (nope_offs // 64).to(tl.int64)
    t_offs = tl.arange(0, BLOCK_T)

    # Cast Q to BF16 once for tensor-core dot
    q_nope_bf16 = q_nope.to(tl.bfloat16)
    q_rope_bf16 = q_rope.to(tl.bfloat16)

    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + t_offs
        t_in_bounds = t_idx < topk
        t_valid = t_idx < valid_topk

        raw_indices = tl.load(
            indices_ptr + bid * stride_ib + t_idx,
            mask=t_in_bounds,
            other=-1,
        )
        idx_valid = t_valid & (raw_indices >= 0)
        safe_indices = tl.where(idx_valid, raw_indices, tl.zeros_like(raw_indices))
        page_ids = (safe_indices // page_size).to(tl.int64)
        page_offs_t = (safe_indices % page_size).to(tl.int64)
        token_data_bases = page_ids * page_bytes + page_offs_t * 576

        # ---- Gather KV nope (FP8) + scales (UE8M0) once for ALL heads in tile ----
        nope_addrs = token_data_bases[:, None] + nope_offs[None, :].to(tl.int64)
        nope_2d_mask = idx_valid[:, None] & nope_mask[None, :]
        kv_nope_fp8 = tl.load(
            cache_fp8_ptr + nope_addrs,
            mask=nope_2d_mask,
            other=0.0,
        )
        scale_bases = page_ids * page_bytes + scale_section_off + page_offs_t * 8
        scale_addrs = scale_bases[:, None] + group_ids[None, :]
        scale_raw = tl.load(
            cache_uint8_ptr + scale_addrs,
            mask=nope_2d_mask,
            other=127,
        )
        scale_f32 = tl.math.exp2(scale_raw.to(tl.float32) - 127.0)
        kv_nope = tl.where(nope_2d_mask, kv_nope_fp8.to(tl.float32) * scale_f32, 0.0)
        kv_nope_bf16 = kv_nope.to(tl.bfloat16)  # [BLOCK_T, NOPE_PAD]

        rope_byte_bases = token_data_bases + 448
        rope_elem_bases = (rope_byte_bases // 2).to(tl.int64)
        rope_addrs = rope_elem_bases[:, None] + rope_offs[None, :].to(tl.int64)
        kv_rope = tl.load(
            cache_bf16_ptr + rope_addrs,
            mask=idx_valid[:, None],
            other=0.0,
        )  # already bf16; keep as bf16 for dot
        kv_rope_bf16 = kv_rope.to(tl.bfloat16)
        kv_rope_f32 = kv_rope.to(tl.float32)

        # ---- Scores via tensor core: [H_TILE, BLOCK_T] ----
        # score = q_nope @ kv_nope.T + q_rope @ kv_rope.T
        scores = tl.dot(q_nope_bf16, tl.trans(kv_nope_bf16))
        scores += tl.dot(q_rope_bf16, tl.trans(kv_rope_bf16))
        scores = tl.where(idx_valid[None, :], scores, -1e30)
        scores = tl.where(h_mask[:, None], scores, -1e30)

        # ---- Per-head online softmax ----
        scores_log2 = scores * LOG2E  # [H_TILE, BLOCK_T]
        tile_max = tl.max(scores_log2, axis=1)  # [H_TILE]
        m_new = tl.maximum(m_i, tile_max)
        alpha = tl.math.exp2(m_i - m_new)  # [H_TILE]
        p = tl.math.exp2(scores_log2 - m_new[:, None])  # [H_TILE, BLOCK_T]
        p = tl.where(idx_valid[None, :], p, 0.0)
        p = tl.where(h_mask[:, None], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # ---- Accumulator update via tensor core ----
        # acc_nope[h, d] += sum_t p[h, t] * kv_nope[t, d]
        p_bf16 = p.to(tl.bfloat16)
        acc_nope = acc_nope * alpha[:, None] + tl.dot(p_bf16, kv_nope_bf16)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p_bf16, kv_rope_bf16)
        m_i = m_new

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc_nope = acc_nope / safe_l[:, None]
    acc_rope = acc_rope / safe_l[:, None]
    lse = tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf"))

    o_base_b = bid * stride_ob
    o_nope_addr = o_base_b + h_offs[:, None] * stride_oh + nope_offs[None, :]
    tl.store(
        O_ptr + o_nope_addr,
        acc_nope.to(tl.bfloat16),
        mask=h_mask[:, None] & nope_mask[None, :],
    )
    o_rope_addr = (
        o_base_b + h_offs[:, None] * stride_oh + NOPE_DIM_RT + rope_offs[None, :]
    )
    tl.store(
        O_ptr + o_rope_addr,
        acc_rope.to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(LSE_ptr + bid * H + h_offs, lse, mask=h_mask)


def run_mhead_sparse_decode(
    q: torch.Tensor,  # [B, 1, H, D] bf16
    k_cache: torch.Tensor,  # [num_pages, page_size, 1, bpt] float8
    indices: torch.Tensor,  # [B, ...] int32
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, H, D = q.shape
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    page_bytes = k_cache.stride(0)

    flat_indices = indices.reshape(B, -1).contiguous()
    topk = flat_indices.shape[1]

    total_elems = num_pages * page_bytes
    raw_flat = k_cache.as_strided((total_elems,), (1,))
    raw_uint8 = raw_flat.view(torch.uint8)
    raw_fp8 = raw_uint8.view(torch.float8_e4m3fn)
    raw_bf16 = raw_uint8.view(torch.bfloat16)

    q3 = q.squeeze(1)
    if not q3.is_contiguous():
        q3 = q3.contiguous()

    out = torch.zeros(B, H, D, dtype=torch.bfloat16, device=q.device)
    lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)

    topk_rounded = triton.next_power_of_2(topk)

    def grid(meta):
        return (B, triton.cdiv(H, meta["H_TILE"]))

    _mhead_tiled_sparse_decode_kernel[grid](
        q3,
        raw_fp8,
        raw_uint8,
        raw_bf16,
        flat_indices,
        (
            topk_length
            if topk_length is not None
            else torch.empty(0, device=q.device, dtype=torch.int32)
        ),
        out,
        lse,
        softmax_scale,
        page_size,
        int(page_bytes),
        int(page_size * _TOKEN_DATA_STRIDE),
        H,
        topk,
        topk_rounded,
        topk_length is not None,
        q3.stride(0),
        q3.stride(1),
        out.stride(0),
        out.stride(1),
        flat_indices.stride(0),
        NOPE_PAD=512,
        ROPE_DIM=_ROPE_DIM,
        NOPE_DIM_RT=_NOPE_DIM,
    )

    return out.unsqueeze(1), lse.unsqueeze(1)
