"""Low-ratio indexer logits for Hopper.

DeepGEMM's fp8_fp4 mqa-logits kernels need SM100/SM120. These Triton kernels
cover the same score math on SM90. Decode reads visible compressed positions
directly from the FP4 indexer pool. Prefill converts each visible K row once
to E4M3, then fuses dot, relu, head weighting, and head reduction without
materializing the much larger per-head score tensor.

Decode follows the BF16 rounding points of the torch reference. Prefill uses an
E4M3 dot with FP32 accumulation and preserves the reference post-dot rounding;
the caller runs the same masking, candidate, and top-k logic on the returned
FP32 logits as the per-request torch loop.
"""

import torch
import triton
import triton.language as tl

INDEX_HEAD_DIM = 128
PAYLOAD_BYTES = tl.constexpr(64)
SCALE_BYTES = tl.constexpr(4)
FP8_E4M3_MAX = 448.0


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


@triton.jit
def _unpack_fp4_index_keys_to_fp8_kernel(
    slots_ptr,
    table_ptr,
    out_ptr,
    page_size,
    row_stride,
    HALF_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Decode block-scaled E2M1 values directly into E4M3."""
    row = tl.program_id(0)
    offs_i = tl.arange(0, HALF_D)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    page = slot // page_size
    off = slot % page_size
    row_base = page * row_stride
    pay = tl.load(table_ptr + row_base + off * PAYLOAD_BYTES + offs_i)
    scale_block = offs_i // 16
    exps = tl.load(
        table_ptr
        + row_base
        + page_size * PAYLOAD_BYTES
        + off * SCALE_BYTES
        + scale_block
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    low = tl.clamp(_e2m1_decode(pay & 0x0F) * scale, -FP8_MAX, FP8_MAX).to(
        out_ptr.dtype.element_ty
    )
    high = tl.clamp(
        _e2m1_decode((pay >> 4) & 0x0F) * scale,
        -FP8_MAX,
        FP8_MAX,
    ).to(out_ptr.dtype.element_ty)
    out = out_ptr + row * (HALF_D * 2)
    tl.store(out + 2 * offs_i, low)
    tl.store(out + 2 * offs_i + 1, high)


def unpack_fp4_index_keys_to_fp8(
    slots: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Gather FP4 index-K rows and decode them directly into E4M3."""
    assert slots.dim() == 1
    assert table.dtype == torch.uint8 and table.dim() == 2
    slots = slots.to(torch.int64).contiguous()
    values = torch.empty(
        (slots.shape[0], INDEX_HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=slots.device,
    )
    if slots.numel() > 0:
        _unpack_fp4_index_keys_to_fp8_kernel[(slots.shape[0],)](
            slots,
            table,
            values,
            page_size,
            table.stride(0),
            HALF_D=INDEX_HEAD_DIM // 2,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=4,
        )
    return values


@triton.jit
def _quantize_bf16_index_queries_fp8_kernel(
    q_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_ob,
    stride_oh,
    H: tl.constexpr,
    D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + row * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :]
    ).to(tl.float32)
    q_fp8 = tl.clamp(q, -FP8_MAX, FP8_MAX).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + row * stride_ob + offs_h[:, None] * stride_oh + offs_d[None, :],
        q_fp8,
    )


def quantize_bf16_index_queries_fp8(
    q: torch.Tensor,
) -> torch.Tensor:
    """Cast each BF16 query head to E4M3 for one K=128 tensor-core dot."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    q = q.contiguous()
    rows, heads, _ = q.shape
    values = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    if rows > 0:
        _quantize_bf16_index_queries_fp8_kernel[(rows,)](
            q,
            values,
            q.stride(0),
            q.stride(1),
            values.stride(0),
            values.stride(1),
            H=heads,
            D=INDEX_HEAD_DIM,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=8,
        )
    return values


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


@triton.jit
def _fp8_index_logits_prefill_kernel(
    q_ptr,  # [B, H, D] e4m3
    w_ptr,  # [B, H] bf16
    k_ptr,  # [L, D] e4m3
    lens_ptr,  # [B] int64
    out_ptr,  # [B, OUT_L] fp32
    L,
    OUT_L,
    stride_qb,
    stride_qh,
    stride_kl,
    stride_wb,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    n_vis = tl.load(lens_ptr + b)
    valid = offs_l < tl.minimum(n_vis, L)
    q = tl.load(q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :])
    k = tl.load(
        k_ptr + offs_l[:, None] * stride_kl + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    acc = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    # Preserve the reference post-dot rounding and reduction points.
    s = acc.to(tl.bfloat16).to(tl.float32)
    s = tl.maximum(s, 0.0)
    w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
    s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    logit = tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)
    logit = tl.where(valid, logit, float("-inf"))
    tl.store(out_ptr + b * OUT_L + offs_l, logit, mask=offs_l < OUT_L)


def fp8_index_logits_prefill(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    lens: torch.Tensor,
) -> torch.Tensor:
    """Score E4M3 queries against E4M3 keys with FP32 accumulation.

    Output rows are padded to four floats for the fused ragged top-k kernel;
    padding remains unreachable because it is initialized to ``-inf``.
    """
    assert q.dtype == torch.float8_e4m3fn and q.shape[-1] == INDEX_HEAD_DIM
    rows, heads, _ = q.shape
    width = keys.shape[0]
    assert keys.dtype == torch.float8_e4m3fn and keys.shape[1:] == (INDEX_HEAD_DIM,)
    assert weights.shape == (rows, heads)
    assert lens.shape == (rows,)
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    keys = keys.contiguous()
    out_width = ((width + 3) // 4) * 4
    out = torch.empty((rows, out_width), dtype=torch.float32, device=q.device)
    if rows == 0 or width == 0:
        return out
    block_l = 64
    _fp8_index_logits_prefill_kernel[(rows, triton.cdiv(out_width, block_l))](
        q,
        weights,
        keys,
        lens.to(torch.int64).contiguous(),
        out,
        width,
        out_width,
        q.stride(0),
        q.stride(1),
        keys.stride(0),
        weights.stride(0),
        H=heads,
        D=INDEX_HEAD_DIM,
        BLOCK_L=block_l,
        num_warps=4,
    )
    return out
