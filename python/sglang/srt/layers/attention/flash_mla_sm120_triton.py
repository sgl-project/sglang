"""SM120-optimized Triton FlashMLA sparse decode kernel — Tiled V2.

Replaces V1's serial token loop with a tiled vectorized approach:
  1. BLOCK_T tokens loaded simultaneously via 2D gather (vs 1-at-a-time)
  2. All BLOCK_T QK scores computed at once via vectorized mul-reduce
  3. V accumulation via vectorized weighted sum across BLOCK_T tokens
  4. Online softmax operates on tile-level maxima (fewer rescales)

Three typed views of the same paged buffer handle FP8/uint8/BF16 regions:
  - float8_e4m3fn view -> nope FP8 values (direct load + dequant)
  - uint8 view         -> UE8M0 scale bytes (raw integer -> exp2 conversion)
  - bfloat16 view      -> rope BF16 values (direct load)

DSv4 page layout (per token, 576 bytes data + 8 bytes scales):
  Data section:  [0:448] FP8 nope | [448:576] BF16 rope (64 values = 128 bytes)
  Scale section: [page_size*576 + offset*8 : +7] UE8M0 scales (7 groups of 64)

Target: RTX PRO 6000 (SM120, 188 SMs, 99KB SMEM, ~1.5 TB/s GDDR7, 96MB L2)
"""

import logging
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

LOG2E = tl.constexpr(1.4426950408889634)

# DSv4 KV cache layout constants
_NOPE_DIM = 448
_ROPE_DIM = 64
_D = _NOPE_DIM + _ROPE_DIM  # 512
_TOKEN_DATA_STRIDE = 576  # bytes per token in data section
_SCALE_STRIDE = 8  # bytes per token in scale section


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 32}, num_warps=8, num_stages=2),
    ],
    key=["topk_rounded"],
)
@triton.jit
def _tiled_sparse_decode_kernel(
    # Q: [B, H, D] bf16
    Q_ptr,
    # Paged KV cache — three typed views of same underlying memory
    cache_fp8_ptr,    # float8_e4m3fn flat (1 byte/elem) — for nope
    cache_uint8_ptr,  # uint8 flat (1 byte/elem) — for scales
    cache_bf16_ptr,   # bfloat16 flat (2 bytes/elem) — for rope
    # Indices: [B, topk] int32
    indices_ptr,
    # Valid lengths: [B] int32
    topk_len_ptr,
    # Output: [B, H, D] bf16 and LSE: [B, H] float32
    O_ptr,
    LSE_ptr,
    # Scalars
    sm_scale: tl.float32,
    page_size: tl.int32,
    page_bytes: tl.int64,
    scale_section_off: tl.int64,  # page_size * 576
    H: tl.int32,
    topk: tl.int32,
    topk_rounded: tl.int32,  # for autotune key
    has_topk_len: tl.constexpr,
    # Strides
    stride_qb: tl.int32,
    stride_qh: tl.int32,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    stride_ib: tl.int32,  # indices batch stride
    # Constexprs
    NOPE_PAD: tl.constexpr,    # 512 (padded from 448)
    ROPE_DIM: tl.constexpr,    # 64
    NOPE_DIM_RT: tl.int32,     # 448 (runtime, for masking)
    BLOCK_T: tl.constexpr,     # tokens per tile (16 or 32)
):
    """Tiled sparse decode: vectorized gather + QK + softmax + V accumulation.

    Grid: (B, H) — one block per (batch, head) pair.
    Each block processes all topk tokens in tiles of BLOCK_T.
    """
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    # ---- Load Q for this (batch, head) ----
    q_base = bid * stride_qb + hid * stride_qh
    nope_offs = tl.arange(0, NOPE_PAD)  # [512]
    nope_mask = nope_offs < NOPE_DIM_RT  # [512], True for [0:448]
    rope_offs = tl.arange(0, ROPE_DIM)  # [64]

    q_nope = tl.load(Q_ptr + q_base + nope_offs, mask=nope_mask, other=0.0)
    q_nope = q_nope.to(tl.float32) * sm_scale
    q_rope = tl.load(Q_ptr + q_base + NOPE_DIM_RT + rope_offs)
    q_rope = q_rope.to(tl.float32) * sm_scale

    # ---- Valid token count ----
    valid_topk = topk
    if has_topk_len:
        valid_topk = tl.load(topk_len_ptr + bid).to(tl.int32)
        valid_topk = tl.minimum(valid_topk, topk)

    # ---- Online softmax state (base-2 math for SM120 efficiency) ----
    m_i: tl.float32 = -1e30
    l_i: tl.float32 = 0.0
    acc_nope = tl.zeros([NOPE_PAD], dtype=tl.float32)
    acc_rope = tl.zeros([ROPE_DIM], dtype=tl.float32)

    # ---- Precompute constant index vectors ----
    group_ids = (nope_offs // 64).to(tl.int64)  # [NOPE_PAD], scale group for each dim
    t_offs = tl.arange(0, BLOCK_T)  # [BLOCK_T], token offsets within tile

    # ---- Process tokens in tiles of BLOCK_T ----
    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + t_offs  # [BLOCK_T], global token indices
        t_in_bounds = t_idx < topk  # bounds for index load
        t_valid = t_idx < valid_topk  # bounds for actual processing

        # Load indices for this tile: [BLOCK_T]
        raw_indices = tl.load(
            indices_ptr + bid * stride_ib + t_idx,
            mask=t_in_bounds, other=-1,
        )
        idx_valid = t_valid & (raw_indices >= 0)  # [BLOCK_T] mask

        # Page addressing: [BLOCK_T] (clamp for safe addressing of invalid tokens)
        safe_indices = tl.where(idx_valid, raw_indices, tl.zeros_like(raw_indices))
        page_ids = (safe_indices // page_size).to(tl.int64)
        page_offs_t = (safe_indices % page_size).to(tl.int64)
        token_data_bases = page_ids * page_bytes + page_offs_t * 576  # [BLOCK_T] int64

        # ---- Vectorized NOPE FP8 gather: [BLOCK_T, NOPE_PAD] ----
        nope_addrs = token_data_bases[:, None] + nope_offs[None, :].to(tl.int64)
        nope_2d_mask = idx_valid[:, None] & nope_mask[None, :]
        kv_nope_fp8 = tl.load(
            cache_fp8_ptr + nope_addrs,
            mask=nope_2d_mask, other=0.0,
        )

        # ---- Vectorized scale gather + dequant: [BLOCK_T, NOPE_PAD] ----
        scale_bases = page_ids * page_bytes + scale_section_off + page_offs_t * 8
        scale_addrs = scale_bases[:, None] + group_ids[None, :]
        scale_raw = tl.load(
            cache_uint8_ptr + scale_addrs,
            mask=nope_2d_mask, other=127,
        )
        scale_f32 = tl.math.exp2(scale_raw.to(tl.float32) - 127.0)
        kv_nope = tl.where(nope_2d_mask, kv_nope_fp8.to(tl.float32) * scale_f32, 0.0)

        # ---- Vectorized ROPE BF16 gather: [BLOCK_T, ROPE_DIM] ----
        rope_byte_bases = token_data_bases + 448
        rope_elem_bases = (rope_byte_bases // 2).to(tl.int64)
        rope_addrs = rope_elem_bases[:, None] + rope_offs[None, :].to(tl.int64)
        kv_rope = tl.load(
            cache_bf16_ptr + rope_addrs,
            mask=idx_valid[:, None], other=0.0,
        ).to(tl.float32)

        # ---- Vectorized QK scores: [BLOCK_T] ----
        # scores[t] = dot(q_nope, kv_nope[t]) + dot(q_rope, kv_rope[t])
        scores = (
            tl.sum(q_nope[None, :] * kv_nope, axis=1)
            + tl.sum(q_rope[None, :] * kv_rope, axis=1)
        )
        scores = tl.where(idx_valid, scores, -1e30)

        # ---- Online softmax update (base-2, tile-level) ----
        scores_log2 = scores * LOG2E  # [BLOCK_T]
        tile_max = tl.max(scores_log2)  # scalar
        m_new = tl.maximum(m_i, tile_max)

        alpha = tl.math.exp2(m_i - m_new)  # rescale factor
        p = tl.math.exp2(scores_log2 - m_new)  # [BLOCK_T] attention weights
        p = tl.where(idx_valid, p, 0.0)  # zero out invalid

        l_i = l_i * alpha + tl.sum(p)

        # ---- Vectorized V accumulation (K=V in MLA) ----
        # acc += sum_t(p[t] * kv[t, :]) for both nope and rope
        acc_nope = acc_nope * alpha + tl.sum(p[:, None] * kv_nope, axis=0)
        acc_rope = acc_rope * alpha + tl.sum(p[:, None] * kv_rope, axis=0)
        m_i = m_new

    # ---- Normalize output ----
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc_nope = acc_nope / safe_l
    acc_rope = acc_rope / safe_l

    # LSE: convert from log2 back to natural log
    lse = tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf"))

    # ---- Store output ----
    o_base = bid * stride_ob + hid * stride_oh
    tl.store(O_ptr + o_base + nope_offs, acc_nope.to(tl.bfloat16), mask=nope_mask)
    tl.store(O_ptr + o_base + NOPE_DIM_RT + rope_offs, acc_rope.to(tl.bfloat16))
    tl.store(LSE_ptr + bid * H + hid, lse)


def _run_triton_sparse_decode(
    q: torch.Tensor,            # [B, 1, H, D] bf16
    k_cache: torch.Tensor,      # [num_pages, page_size, 1, bpt] float8
    indices: torch.Tensor,      # [B, ...] int32
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the tiled Triton sparse decode kernel on one paged KV cache."""
    B, _, H, D = q.shape
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    page_bytes = k_cache.stride(0)  # elements = bytes for float8

    # Flatten indices to [B, topk]
    flat_indices = indices.reshape(B, -1).contiguous()
    topk = flat_indices.shape[1]

    # Create three typed views of the flat cache memory
    total_elems = num_pages * page_bytes
    raw_fp8 = k_cache.as_strided((total_elems,), (1,))
    raw_uint8 = raw_fp8.view(torch.uint8)
    raw_bf16 = raw_uint8.view(torch.bfloat16)

    # Squeeze Q: [B, H, D]
    q3 = q.squeeze(1)
    if not q3.is_contiguous():
        q3 = q3.contiguous()

    out = torch.zeros(B, H, D, dtype=torch.bfloat16, device=q.device)
    lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)

    # Round topk for autotune key stability
    topk_rounded = triton.next_power_of_2(topk)

    grid = (B, H)
    _tiled_sparse_decode_kernel[grid](
        q3,
        raw_fp8, raw_uint8, raw_bf16,
        flat_indices,
        topk_length if topk_length is not None else torch.empty(
            0, device=q.device, dtype=torch.int32
        ),
        out, lse,
        softmax_scale,
        page_size,
        int(page_bytes),  # page_bytes (int64)
        int(page_size * _TOKEN_DATA_STRIDE),  # scale_section_off (int64)
        H, topk, topk_rounded,
        topk_length is not None,
        q3.stride(0), q3.stride(1),
        out.stride(0), out.stride(1),
        flat_indices.stride(0),
        NOPE_PAD=512,
        ROPE_DIM=_ROPE_DIM,
        NOPE_DIM_RT=_NOPE_DIM,
    )

    # Return [B, 1, H, D] and [B, 1, H]
    return out.unsqueeze(1), lse.unsqueeze(1)


def _merge_partial_attn(
    out1: torch.Tensor, lse1: torch.Tensor,
    out2: torch.Tensor, lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention outputs using LSE-weighted combination.

    out: [B, 1, H, D] bf16, lse: [B, 1, H] float32
    """
    max_lse = torch.maximum(lse1, lse2)
    w1 = torch.where(lse1 > -1e20, torch.exp(lse1 - max_lse), torch.zeros_like(lse1))
    w2 = torch.where(lse2 > -1e20, torch.exp(lse2 - max_lse), torch.zeros_like(lse2))
    total = (w1 + w2).clamp(min=1e-20)
    merged = (
        w1.unsqueeze(-1) * out1.float() + w2.unsqueeze(-1) * out2.float()
    ) / total.unsqueeze(-1)
    merged_lse = max_lse + torch.log(total)
    return merged.to(torch.bfloat16), merged_lse


def _apply_attn_sink(
    out: torch.Tensor, lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply attention sink normalization.

    The sink adds to the softmax denominator without contributing output,
    effectively down-weighting all attention scores.

    out: [B, 1, H, D] bf16, lse: [B, 1, H] f32, attn_sink: [H] f32
    """
    sink_lse = attn_sink.view(1, 1, -1).expand_as(lse)
    combined_lse = torch.logaddexp(lse, sink_lse)
    w = torch.where(
        lse > -1e20,
        torch.exp(lse - combined_lse),
        torch.zeros_like(lse),
    )
    return (out.float() * w.unsqueeze(-1)).to(torch.bfloat16), combined_lse


def flash_mla_sparse_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
    softmax_scale: float,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SM120-optimized sparse MLA decode using tiled Triton kernel.

    Processes SWA and extra (c4/c128) caches separately via the same
    Triton kernel, then merges results using LSE-weighted combination.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # Process main cache (SWA)
    out, lse = _run_triton_sparse_decode(
        q, k_cache, indices, topk_length, softmax_scale,
    )

    # Process extra cache (c4 / c128) if present
    if extra_k_cache is not None and extra_indices is not None:
        out_extra, lse_extra = _run_triton_sparse_decode(
            q, extra_k_cache, extra_indices, extra_topk_length, softmax_scale,
        )
        out, lse = _merge_partial_attn(out, lse, out_extra, lse_extra)

    # Apply attention sink
    if attn_sink is not None:
        out, lse = _apply_attn_sink(out, lse, attn_sink)

    # Return format matching PyTorch fallback: (out, lse.permute(0,2,1))
    return out, lse.permute(0, 2, 1)
