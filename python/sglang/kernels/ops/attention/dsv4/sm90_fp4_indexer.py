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
    slots_ptr,  # [B, L] pool slots per (request, compressed position)
    req_to_token_ptr,  # [num_reqs, max_context_len] full-token pool slots
    req_ptr,  # [B] request-pool row for each query
    lens_ptr,  # [B] int64 visible compressed positions per request
    table_ptr,  # [num_pages, page_size * 64 + page_size * 4] uint8
    out_ptr,  # [B, L] fp32 logits, -inf beyond lens
    candidate_scores_ptr,
    candidate_lens_ptr,
    L,
    page_size,
    row_stride,
    stride_qb,
    stride_qh,
    stride_wb,
    stride_req,
    stride_out,
    candidate_score_stride,
    H: tl.constexpr,
    HALF_D: tl.constexpr,  # D // 2 == 64 nibble-pairs per row
    BLOCK_L: tl.constexpr,
    RATIO: tl.constexpr,
    USE_REQ_TO_TOKEN: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
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
    if USE_REQ_TO_TOKEN:
        req = tl.load(req_ptr + b).to(tl.int64)
        slot = tl.load(
            req_to_token_ptr + req * stride_req + offs_l * RATIO,
            mask=valid,
            other=0,
        ).to(tl.int64)
        slot = slot // RATIO
    else:
        slot = tl.load(slots_ptr + b * L + offs_l, mask=offs_l < L, other=0).to(
            tl.int64
        )
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
    tl.store(out_ptr + b * stride_out + offs_l, logit, mask=offs_l < L)
    if WRITE_CANDIDATES:
        blocks_per_tile: tl.constexpr = BLOCK_L // CANDIDATE_BLOCK_SIZE
        block_scores = tl.reshape(logit, (blocks_per_tile, CANDIDATE_BLOCK_SIZE))
        block_scores = tl.max(block_scores, axis=1)
        block_ids = lb * blocks_per_tile + tl.arange(0, blocks_per_tile)
        last_block = (n_vis - 1) // CANDIDATE_BLOCK_SIZE
        block_scores = tl.where(
            (n_vis > 0) & (block_ids == last_block), float("inf"), block_scores
        )
        num_blocks = (L + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
        tl.store(
            candidate_scores_ptr + b * candidate_score_stride + block_ids,
            block_scores,
            mask=block_ids < num_blocks,
        )
        tl.store(
            candidate_lens_ptr + b,
            (n_vis + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
            mask=lb == 0,
        )


def fp4_index_logits_decode(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """q [B, H, 128] bf16, weights [B, H], slots [B, L], lens [B] int64,
    table = the layer's fp4 index-K page buffer (uint8, 2D). Returns [B, L] fp32
    logits with -inf at positions >= lens."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    B, H, _ = q.shape
    L = slots.shape[1]
    assert table.dtype == torch.uint8 and table.dim() == 2
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    slots = slots.contiguous()
    out_storage = torch.empty(
        (B, triton.cdiv(L, 4) * 4), dtype=torch.float32, device=q.device
    )
    out = out_storage[:, :L]
    if L == 0:
        return out
    BLOCK_L = 64
    grid = (B, triton.cdiv(L, BLOCK_L))
    _fp4_index_logits_kernel[grid](
        q,
        weights,
        slots,
        slots,
        lens,
        lens.to(torch.int64).contiguous(),
        table,
        out,
        out,
        lens,
        L,
        page_size,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        slots.stride(0),
        out.stride(0),
        out.stride(0),
        H=H,
        HALF_D=INDEX_HEAD_DIM // 2,
        BLOCK_L=BLOCK_L,
        RATIO=1,
        USE_REQ_TO_TOKEN=False,
        CANDIDATE_BLOCK_SIZE=1,
        WRITE_CANDIDATES=False,
        num_warps=4,
    )
    return out


def fp4_index_logits_req_to_token(
    q: torch.Tensor,
    weights: torch.Tensor,
    req_to_token: torch.Tensor,
    req: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    ratio: int,
    width: int,
    candidate_block_size: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score logical compressed positions without materializing their pool slots."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    B, H, _ = q.shape
    assert req.shape == lens.shape == (B,)
    assert req_to_token.dim() == 2
    assert table.dtype == torch.uint8 and table.dim() == 2
    assert ratio in (1, 2)
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    req = req.to(torch.int64).contiguous()
    lens = lens.to(torch.int64).contiguous()
    out_storage = torch.empty(
        (B, triton.cdiv(width, 4) * 4), dtype=torch.float32, device=q.device
    )
    out = out_storage[:, :width]
    block_l = 64
    if candidate_block_size:
        assert block_l % candidate_block_size == 0
        num_blocks = triton.cdiv(width, candidate_block_size)
        candidate_storage = torch.empty(
            (B, triton.cdiv(num_blocks, 4) * 4),
            dtype=torch.float32,
            device=q.device,
        )
        candidate_scores = candidate_storage[:, :num_blocks]
        candidate_lens = torch.empty(B, dtype=torch.int32, device=q.device)
    else:
        candidate_scores = out
        candidate_lens = lens
    if width == 0:
        if candidate_block_size:
            return out, candidate_scores, candidate_lens
        return out
    grid = (B, triton.cdiv(width, block_l))
    _fp4_index_logits_kernel[grid](
        q,
        weights,
        req_to_token,
        req_to_token,
        req,
        lens,
        table,
        out,
        candidate_scores,
        candidate_lens,
        width,
        page_size,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        req_to_token.stride(0),
        out.stride(0),
        candidate_scores.stride(0),
        H=H,
        HALF_D=INDEX_HEAD_DIM // 2,
        BLOCK_L=block_l,
        RATIO=ratio,
        USE_REQ_TO_TOKEN=True,
        CANDIDATE_BLOCK_SIZE=candidate_block_size or 1,
        WRITE_CANDIDATES=bool(candidate_block_size),
        num_warps=4,
    )
    if candidate_block_size:
        return out, candidate_scores, candidate_lens
    return out
