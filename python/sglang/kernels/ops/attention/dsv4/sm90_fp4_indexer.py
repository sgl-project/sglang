"""Decode-time low-ratio indexer logits for Hopper.

DeepGEMM's fp8_fp4 mqa-logits kernels need SM100/SM120. This Triton kernel covers
the same decode step on SM90: one query token per request, scored against the
request's visible compressed positions read straight out of the fp4 indexer
pool (e2m1 payload + e8m0 per-32 block scales, page layout of
store_fp4_index_k_cache), summed over heads with relu and the per-head weights.

Numerics follow the torch reference path (bf16 dot, bf16 relu/weight product,
bf16 head reduction); the caller runs the same masking / candidate / top-k
logic on the returned fp32 logits as the per-request torch loop.
"""

import torch
import triton
import triton.language as tl

INDEX_HEAD_DIM = 128
PAYLOAD_BYTES = tl.constexpr(64)
SCALE_BYTES = tl.constexpr(4)


@triton.jit
def _e2m1_decode(code):
    # code: uint 0..15 -> e2m1 value. exp = bits 2..1, mantissa = bit 0, sign = bit 3.
    e = (code >> 1) & 3
    m = (code & 1).to(tl.float32)
    sub = m * 0.5
    nor = (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32))
    v = tl.where(e == 0, sub, nor)
    return tl.where((code >> 3) == 1, -v, v)


@triton.jit
def _fp4_index_logits_kernel(
    q_ptr,  # [B, H, D] bf16, fq4 queries (already rope'd)
    w_ptr,  # [B, H] bf16 head weights (softmax scale folded in)
    slots_ptr,  # [B, L] int64 pool slots per (request, compressed position)
    lens_ptr,  # [B] int64 visible compressed positions per request
    table_ptr,  # [num_pages, page_size * 64 + page_size * 4] uint8
    out_ptr,  # [B, L] fp32 logits, -inf beyond lens
    L,
    page_size,
    row_stride,
    stride_qb,
    stride_qh,
    stride_wb,
    H: tl.constexpr,
    HALF_D: tl.constexpr,  # D // 2 == 64 nibble-pairs per row
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(
        0, HALF_D
    )  # byte index i holds elements 2i (low nibble), 2i+1 (high nibble)

    n_vis = tl.load(lens_ptr + b)
    valid = offs_l < tl.minimum(n_vis, L)
    slot = tl.load(slots_ptr + b * L + offs_l, mask=offs_l < L, other=0).to(tl.int64)
    page = slot // page_size
    off = slot % page_size
    row_base = page * row_stride

    # K payload: [BLOCK_L, HALF_D] uint8
    pay = tl.load(
        table_ptr + row_base[:, None] + off[:, None] * PAYLOAD_BYTES + offs_i[None, :],
        mask=valid[:, None],
        other=0,
    )
    low = _e2m1_decode(pay & 0x0F)
    high = _e2m1_decode((pay >> 4) & 0x0F)
    # e8m0 block scales: element j uses block j // 32 -> byte i uses block i // 16.
    sc_idx = offs_i // 16
    exps = tl.load(
        table_ptr
        + row_base[:, None]
        + page_size * PAYLOAD_BYTES
        + off[:, None] * SCALE_BYTES
        + sc_idx[None, :],
        mask=valid[:, None],
        other=127,
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    k_low = (low * scale).to(tl.bfloat16)  # [BLOCK_L, HALF_D] elements 2i
    k_high = (high * scale).to(tl.bfloat16)  # elements 2i+1

    # queries: even / odd elements, [H, HALF_D] bf16
    q_even = tl.load(
        q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + 2 * offs_i[None, :]
    )
    q_odd = tl.load(
        q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + 2 * offs_i[None, :] + 1
    )

    acc = tl.dot(q_even, tl.trans(k_low))  # [H, BLOCK_L] fp32
    acc += tl.dot(q_odd, tl.trans(k_high))
    # reference rounding points: bf16 dot -> relu -> * bf16 weight -> bf16 -> sum -> bf16
    s = acc.to(tl.bfloat16).to(tl.float32)
    s = tl.maximum(s, 0.0)
    w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
    s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    logit = tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)
    logit = tl.where(valid, logit, float("-inf"))
    tl.store(out_ptr + b * L + offs_l, logit, mask=offs_l < L)


def fp4_index_logits_decode(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """q [B, H, 128] bf16, weights [B, H], slots [B, L] int64, lens [B] int64,
    table = the layer's fp4 index-K page buffer (uint8, 2D). Returns [B, L] fp32
    logits with -inf at positions >= lens."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    B, H, _ = q.shape
    L = slots.shape[1]
    assert table.dtype == torch.uint8 and table.dim() == 2
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    slots = slots.contiguous()
    out = torch.empty((B, L), dtype=torch.float32, device=q.device)
    if L == 0:
        return out
    BLOCK_L = 64
    grid = (B, triton.cdiv(L, BLOCK_L))
    _fp4_index_logits_kernel[grid](
        q,
        weights,
        slots,
        lens.to(torch.int64).contiguous(),
        table,
        out,
        L,
        page_size,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        H=H,
        HALF_D=INDEX_HEAD_DIM // 2,
        BLOCK_L=BLOCK_L,
        num_warps=4,
    )
    return out
