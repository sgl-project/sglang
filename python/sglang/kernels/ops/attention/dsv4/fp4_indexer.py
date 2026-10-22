from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.jit.utils import aligned_new_empty
from sglang.kernels.ops.attention.deep_select import get_input_stride_alignment_bytes
from sglang.kernels.ops.quantization.fp4_fake_quant import FP4_AMAX_FLOOR
from sglang.kernels.ops.quantization.fp4_indexer_quant import (
    _ceil_ue8m0_exp,
    _fp4_e2m1_code_rne,
)

INDEX_HEAD_DIM = 128
# One index-K slot: 64 packed e2m1 bytes and four ue8m0 block exponents.
INDEX_K_PAYLOAD_BYTES = tl.constexpr(64)
INDEX_K_SCALE_BYTES = tl.constexpr(4)
INDEX_K_SLOT_BYTES = INDEX_K_PAYLOAD_BYTES.value + INDEX_K_SCALE_BYTES.value


@triton.jit
def _finish_paged_indexer_topk_kernel(
    Indices,
    Scores,
    Lengths,
    Requests,
    ReqTable,
    Pages,
    Raw,
    BATCH: tl.constexpr,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    IDX_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    MASK_SCORES: tl.constexpr,
    WRITE_RAW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    idx = tl.load(Indices + row * IDX_STRIDE + col, (row < BATCH) & (col < K), WIDTH)
    if MASK_SCORES:
        valid = (row < BATCH) & (col < K) & (idx >= 0) & (idx < WIDTH)
        score = tl.load(Scores + row * SCORE_STRIDE + idx, valid, -float("inf"))
        idx = tl.where(valid & (score > -float("inf")), idx, WIDTH)
    idx = tl.sort(tl.where(idx >= 0, idx, WIDTH), descending=False)
    length = tl.load(Lengths + row, row < BATCH, 0)
    req = tl.load(Requests + row, row < BATCH, 0)
    valid = (row < BATCH) & (col < K) & (idx < length)
    slot = tl.load(
        ReqTable + req.to(tl.int64) * TABLE_STRIDE + idx.to(tl.int64) * RATIO,
        valid,
        0,
    )
    tl.store(
        Pages + row * PAGE_STRIDE + col,
        tl.where(valid, slot.to(tl.int64) // RATIO, -1),
        col < OUT_WIDTH,
    )
    if WRITE_RAW:
        tl.store(
            Raw + row * RAW_STRIDE + col, tl.where(valid, idx, -1), col < OUT_WIDTH
        )


def finish_paged_indexer_topk(
    indices: torch.Tensor,
    scores: torch.Tensor,
    lengths: torch.Tensor,
    req: torch.Tensor,
    req_table: torch.Tensor,
    page_indices: torch.Tensor,
    raw_indices: torch.Tensor | None,
    ratio: int,
    mask_scores: bool,
) -> None:
    """Sort selected positions and map them to compressed KV slots.

    Candidate consumers discard selected masked scores, including top-k underfill.
    The entire output (including padded rows/columns) is written, with -1 padding.
    """
    if not page_indices.shape[0]:
        return
    _finish_paged_indexer_topk_kernel[(page_indices.shape[0],)](
        indices,
        scores,
        lengths,
        req,
        req_table,
        page_indices,
        raw_indices,
        lengths.numel(),
        indices.shape[1],
        scores.shape[1],
        page_indices.shape[1],
        indices.stride(0),
        scores.stride(0),
        req_table.stride(0),
        page_indices.stride(0),
        raw_indices.stride(0) if raw_indices is not None else 0,
        ratio,
        mask_scores,
        raw_indices is not None,
        triton.next_power_of_2(max(page_indices.shape[1], indices.shape[1])),
        num_warps=4,
    )


# Visible key counts change across batches; keep them out of the JIT key.
@triton.jit(do_not_specialize=["NUM_SLOTS"])
def _finish_flat_indexer_topk_kernel(
    Indices,
    Slots,
    Starts,
    Pages,
    Raw,
    K: tl.constexpr,
    NUM_SLOTS,
    IDX_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    WRITE_RAW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    unselected = 2147483647
    idx = tl.load(Indices + row * IDX_STRIDE + col, col < K, unselected)
    idx = tl.sort(tl.where(idx < 0, unselected, idx), descending=False)
    chosen = (col < K) & (idx != unselected)
    slot = tl.load(Slots + tl.minimum(idx, NUM_SLOTS - 1), chosen, -1)
    tl.store(Pages + row * PAGE_STRIDE + col, slot, col < K)
    if WRITE_RAW:
        start = tl.load(Starts + row)
        tl.store(
            Raw + row * RAW_STRIDE + col, tl.where(chosen, idx - start, -1), col < K
        )


def finish_flat_indexer_topk(
    indices: torch.Tensor,
    slots: torch.Tensor,
    request_starts: torch.Tensor,
    page_indices: torch.Tensor,
    raw_indices: torch.Tensor | None,
) -> None:
    """Sort flattened prefill positions and map them to KV slots in one kernel.

    Negative positions and INT32_MAX are unselected. Write only the selected
    rows and columns; callers retain responsibility for resetting output padding.
    """
    rows, k = indices.shape
    if not rows or not k:
        return
    assert indices.dtype == torch.int32 and indices.stride(1) == 1
    assert slots.ndim == 1 and slots.is_contiguous() and slots.numel() > 0
    assert request_starts.shape == (rows,) and request_starts.is_contiguous()
    assert page_indices.shape[0] >= rows and page_indices.shape[1] >= k
    assert page_indices.dtype == torch.int32 and page_indices.stride(1) == 1
    if raw_indices is not None:
        assert raw_indices.shape[0] >= rows and raw_indices.shape[1] >= k
        assert raw_indices.dtype == torch.int32 and raw_indices.stride(1) == 1
    _finish_flat_indexer_topk_kernel[(rows,)](
        indices,
        slots,
        request_starts,
        page_indices,
        raw_indices,
        k,
        slots.numel(),
        indices.stride(0),
        page_indices.stride(0),
        raw_indices.stride(0) if raw_indices is not None else 0,
        raw_indices is not None,
        triton.next_power_of_2(k),
        num_warps=4,
    )


@triton.jit
def _index_k_rope_pack_kernel(
    X,
    F,
    Pos,
    Payload,
    Scale,
    Cache,
    Loc,
    F_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    RD: tl.constexpr,
    INDEXED: tl.constexpr,
    STORE_CACHE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HEADS
    frow = tl.load(Pos + token) if INDEXED else token
    offsets = tl.arange(0, 128)
    value = tl.load(X + row * 128 + offsets).to(tl.float32)
    tail = offsets >= 128 - RD
    pair = (offsets - (128 - RD)) // 2
    imaginary = (offsets % 2) == 1
    real = tl.load(X + row * 128 + 128 - RD + 2 * pair, tail, 0).to(tl.float32)
    imag = tl.load(X + row * 128 + 128 - RD + 2 * pair + 1, tail, 0).to(tl.float32)
    fr = tl.load(F + frow * F_STRIDE + 2 * pair, tail, 1.0)
    fi = tl.load(F + frow * F_STRIDE + 2 * pair + 1, tail, 0.0)
    rotated = tl.where(imaginary, real * fi + imag * fr, real * fr - imag * fi)
    value = tl.where(tail, rotated.to(tl.bfloat16).to(tl.float32), value)

    blocks = tl.reshape(value, (4, 32))
    amax = tl.maximum(tl.max(tl.abs(blocks), 1), AMAX_FLOOR) * (1.0 / 6.0)
    bits = amax.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    fake_scale = (exponent << 23).to(tl.float32, bitcast=True)
    scaled = tl.minimum(tl.maximum(blocks / fake_scale[:, None], -6.0), 6.0)
    magnitude = tl.abs(scaled)
    step = tl.where(magnitude < 2.0, 0.5, tl.where(magnitude < 4.0, 1.0, 2.0))
    sign = tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))
    rounded = libdevice.rint(magnitude / step) * step * sign
    # Preserve the BF16 intermediate before recomputing the indexer scale.
    dequantized = (rounded * fake_scale[:, None]).to(tl.bfloat16).to(tl.float32)
    pack_amax = tl.max(tl.abs(dequantized), 1)
    pack_exponent = _ceil_ue8m0_exp(tl.maximum(pack_amax / 6.0, 1.0e-4))
    pack_scale = (pack_exponent << 23).to(tl.float32, bitcast=True)
    codes = _fp4_e2m1_code_rne(dequantized / pack_scale[:, None])
    low, high = tl.split(tl.reshape(codes, (64, 2)))
    payload = low | (high << 4)
    sf = tl.sum(pack_exponent.to(tl.uint32) << (tl.arange(0, 4) * 8), 0)
    byte_offsets = tl.arange(0, 64)
    if STORE_CACHE:
        location = tl.load(Loc + token)
        page = location // PAGE_SIZE
        slot = location % PAGE_SIZE
        tl.store(Cache + page * CACHE_STRIDE + slot * 64 + byte_offsets, payload)
        scale_bytes = (sf >> (tl.arange(0, 4) * 8)) & 0xFF
        tl.store(
            Cache + page * CACHE_STRIDE + PAGE_SIZE * 64 + slot * 4 + tl.arange(0, 4),
            scale_bytes,
        )
    else:
        tl.store(Payload + row * 64 + byte_offsets, payload)
        tl.store(Scale + row, sf)


def index_k_rope_pack(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rope_dim: int,
    *,
    positions: torch.Tensor | None = None,
    cache: torch.Tensor | None = None,
    loc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """RoPE, fake fp4 quantization and the indexer pack in one launch: packed
    ``[T*heads, 64]`` / ``[T*heads]`` when ``cache`` is None, else the paged index-K
    cache write. Without positions, freqs is already gathered per token; otherwise
    the kernel reads freqs[positions] directly, removing the gather launch.

    Both quantization stages stay: the indexer packer has a different scale floor
    from fake_quant_fp4, so packing the first stage directly is not equivalent.
    """
    assert x.dtype == torch.bfloat16 and x.shape[-1] == 128
    assert 0 <= rope_dim <= 128 and rope_dim % 2 == 0
    x = x.contiguous()
    rows = x.numel() // 128
    heads = x[0].numel() // 128 if x.shape[0] else 1
    f = torch.view_as_real(freqs.contiguous())
    if cache is None:
        payload = torch.empty((rows, 64), dtype=torch.int8, device=x.device)
        scale = torch.empty((rows,), dtype=torch.int32, device=x.device)
        page_size = cache_stride = 0
    else:
        assert heads == 1 and loc is not None and loc.numel() == rows
        assert cache.ndim == 2 and cache.shape[1] % INDEX_K_SLOT_BYTES == 0
        payload = scale = None
        page_size, cache_stride = cache.shape[1] // INDEX_K_SLOT_BYTES, cache.stride(0)
    if rows:
        _index_k_rope_pack_kernel[(rows,)](
            x,
            f,
            positions,
            payload,
            scale,
            cache,
            loc,
            F_STRIDE=f.stride(0),
            HEADS=heads,
            RD=rope_dim,
            INDEXED=positions is not None,
            STORE_CACHE=cache is not None,
            PAGE_SIZE=page_size,
            CACHE_STRIDE=cache_stride,
            AMAX_FLOOR=FP4_AMAX_FLOOR,
            num_warps=4,
        )
    return (payload, scale) if cache is None else None


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
def _fp4_index_logits_tile(
    q_ptr,
    w_ptr,
    table_ptr,
    b,
    slot,
    valid,
    page_size,
    row_stride,
    stride_qb,
    stride_qh,
    stride_wb,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
):
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(0, HALF_D)
    page = slot // page_size
    off = slot % page_size
    row_base = page * row_stride
    # K payload: [BLOCK_L, HALF_D] uint8
    pay = tl.load(
        table_ptr
        + row_base[:, None]
        + off[:, None] * INDEX_K_PAYLOAD_BYTES
        + offs_i[None, :],
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
        + page_size * INDEX_K_PAYLOAD_BYTES
        + off[:, None] * INDEX_K_SCALE_BYTES
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
    return logit


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
    # The caller returns for L == 0 and makes q contiguous. Exclude singleton
    # heads, whose stride is not constrained by PyTorch contiguity.
    tl.assume(L > 0)
    if H > 1:
        tl.assume(stride_qh == HALF_D * 2)
    b = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    n_vis = tl.load(lens_ptr + b)
    # Graph replay keeps the capacity-sized grid even for short live contexts.
    # Skip whole invisible tiles using the current device-side length; masking
    # only the K loads would still run dequantization, dot products and reduction.
    if lb * BLOCK_L >= n_vis:
        tl.store(out_ptr + b * L + offs_l, float("-inf"), mask=offs_l < L)
    else:
        valid = offs_l < tl.minimum(n_vis, L)
        slot = tl.load(slots_ptr + b * L + offs_l, mask=offs_l < L, other=0).to(
            tl.int64
        )
        logit = _fp4_index_logits_tile(
            q_ptr,
            w_ptr,
            table_ptr,
            b,
            slot,
            valid,
            page_size,
            row_stride,
            stride_qb,
            stride_qh,
            stride_wb,
            H,
            HALF_D,
        )
        tl.store(out_ptr + b * L + offs_l, logit, mask=offs_l < L)


def fp4_index_logits_decode(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Decode index logits from the fp4 index-K pool. q [B, H, 128] bf16, weights
    [B, H], slots [B, L] int64, lens [B] int64, table = the layer's index-K page
    buffer (uint8, 2D). Returns [B, L] fp32 logits, -inf at positions >= lens,
    rounded as the torch reference does."""
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
def _fp4_index_logits_paged_kernel(
    q_ptr,
    w_ptr,
    req_ptr,
    req_table_ptr,
    lens_ptr,
    table_ptr,
    mask_ptr,
    out_ptr,
    L,
    page_size,
    row_stride,
    req_stride,
    mask_stride,
    out_stride,
    stride_qb,
    stride_qh,
    stride_wb,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    n_vis = tl.minimum(tl.load(lens_ptr + b), L)
    req = tl.load(req_ptr + b).to(tl.int64)
    for tile in range(tl.program_id(1), tl.cdiv(n_vis, BLOCK_L), tl.num_programs(1)):
        offs_l = tile * BLOCK_L + tl.arange(0, BLOCK_L)
        valid = offs_l < n_vis
        if HAS_MASK:
            valid = valid & tl.load(
                mask_ptr + b * mask_stride + offs_l, mask=offs_l < n_vis, other=False
            )
        slot = (
            tl.load(
                req_table_ptr + req * req_stride + offs_l * RATIO, mask=valid, other=0
            ).to(tl.int64)
            // RATIO
        )
        logit = _fp4_index_logits_tile(
            q_ptr,
            w_ptr,
            table_ptr,
            b,
            slot,
            valid,
            page_size,
            row_stride,
            stride_qb,
            stride_qh,
            stride_wb,
            H,
            HALF_D,
        )
        tl.store(out_ptr + b * out_stride + offs_l, logit, mask=offs_l < L)


def fp4_index_logits_paged(
    q: torch.Tensor,
    weights: torch.Tensor,
    req: torch.Tensor,
    req_table: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    max_len: int,
    ratio: int,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score visible compressed positions directly from the request table.

    The output has graph-stable capacity, but only the first ``lens[b]`` values
    are initialized. Consumers must use the current device-side lengths.
    The FP4 decoding and BF16 rounding are shared with the dense-slot path.
    """
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    assert table.dtype == torch.uint8 and table.ndim == 2
    assert ratio in (1, 2) and max_len <= req_table.shape[1] // ratio
    assert req_table.stride(1) == 1
    if candidate_mask is not None:
        assert candidate_mask.shape[1] >= max_len and candidate_mask.stride(1) == 1
    batch, heads, _ = q.shape
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    req = req.contiguous()
    lens = lens.contiguous()
    # Rows aligned for DeepSelect top-k (128B) also suit candidate-amax (8 floats).
    out = aligned_new_empty(
        shape=(batch, max_len),
        dtype=torch.float32,
        device=q.device,
        alignment_bytes=get_input_stride_alignment_bytes(),
    )
    if not batch or not max_len:
        return out
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    workers = min(triton.cdiv(max_len, 64), triton.cdiv(num_sms * 4, batch))
    _fp4_index_logits_paged_kernel[(batch, workers)](
        q,
        weights,
        req,
        req_table,
        lens,
        table,
        candidate_mask,
        out,
        max_len,
        page_size,
        table.stride(0),
        req_table.stride(0),
        candidate_mask.stride(0) if candidate_mask is not None else 0,
        out.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        H=heads,
        HALF_D=INDEX_HEAD_DIM // 2,
        RATIO=ratio,
        HAS_MASK=candidate_mask is not None,
        BLOCK_L=64,
        num_warps=4,
    )
    return out
