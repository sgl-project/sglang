"""Fused request-to-KV mapping for opt-in SM90 32-head static verification."""

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
    exp = tl.where(e == 1, 1.0, tl.where(e == 2, 2.0, 4.0))
    nor = (1.0 + m * 0.5) * exp
    v = tl.where(e == 0, sub, nor)
    return tl.where((code >> 3) == 1, -v, v)


@triton.jit
def _fp4_index_logits_kernel(
    q_ptr,  # [B, H, D] bf16, fq4 queries (already rope'd)
    w_ptr,  # [B, H] bf16 head weights (softmax scale folded in)
    slots_ptr,  # [requests, capacity * ratio] int32 request-to-token mapping
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
    req_ptr,
    req_stride: tl.constexpr,
    ratio: tl.constexpr,
):
    b = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(
        0, HALF_D
    )  # byte index i holds elements 2i (low nibble), 2i+1 (high nibble)

    n_vis = tl.load(lens_ptr + b)
    # Keep capacity-sized CUDA Graph buffers while skipping invisible tiles,
    # including mapping loads, dequantization, dot products and head reduction.
    if lb * BLOCK_L >= n_vis:
        tl.store(out_ptr + b * L + offs_l, float("-inf"), mask=offs_l < L)
    else:
        valid = offs_l < tl.minimum(n_vis, L)
        request = tl.load(req_ptr + b).to(tl.int64)
        slot = (
            tl.load(
                slots_ptr + request * req_stride + offs_l * ratio,
                mask=valid,
                other=0,
            ).to(tl.int64)
            // ratio
        )
        page = slot // page_size
        off = slot % page_size
        row_base = page * row_stride

        # K payload: [BLOCK_L, HALF_D] uint8
        pay = tl.load(
            table_ptr
            + row_base[:, None]
            + off[:, None] * PAYLOAD_BYTES
            + offs_i[None, :],
            mask=valid[:, None],
            other=0,
        )
        low = _e2m1_decode(pay & 0x0F)
        high = _e2m1_decode((pay >> 4) & 0x0F)
        # e8m0 block scales: each scale covers 16 payload bytes (32 elements).
        offs_s = tl.arange(0, SCALE_BYTES)
        exps = tl.load(
            table_ptr
            + row_base[:, None]
            + page_size * PAYLOAD_BYTES
            + off[:, None] * SCALE_BYTES
            + offs_s[None, :],
            mask=valid[:, None],
            other=127,
        )
        scale = tl.exp2(exps.to(tl.float32) - 127.0)
        scale = scale[:, :, None]
        k_shape: tl.constexpr = (BLOCK_L, SCALE_BYTES, HALF_D // SCALE_BYTES)
        k_low = tl.reshape(tl.reshape(low, k_shape) * scale, (BLOCK_L, HALF_D)).to(
            tl.bfloat16
        )
        k_high = tl.reshape(tl.reshape(high, k_shape) * scale, (BLOCK_L, HALF_D)).to(
            tl.bfloat16
        )

        # queries: even / odd elements, [H, HALF_D] bf16
        q_even = tl.load(
            q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + 2 * offs_i[None, :]
        )
        q_odd = tl.load(
            q_ptr
            + b * stride_qb
            + offs_h[:, None] * stride_qh
            + 2 * offs_i[None, :]
            + 1
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


def fp4_index_logits_mapped_sm90(
    q,
    weights,
    req_to_token,
    req,
    lens,
    table,
    page_size,
    ratio,
    width,
):
    """Score with request mapping fused into the Triton kernel.

    The caller validates the common tensor/device contract. No dense slots
    tensor or host read of dynamic lengths is needed during graph replay.
    """
    rows, heads, _ = q.shape
    out = torch.empty((rows, width), dtype=torch.float32, device=q.device)
    if rows == 0 or width == 0:
        return out
    _fp4_index_logits_kernel[(rows, triton.cdiv(width, 64))](
        q,
        weights,
        req_to_token,
        lens,
        table,
        out,
        width,
        page_size,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        H=heads,
        HALF_D=64,
        BLOCK_L=64,
        req_ptr=req,
        req_stride=req_to_token.stride(0),
        ratio=ratio,
        num_warps=4,
    )
    return out
