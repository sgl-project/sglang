from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.attention.dsv4.torch_quant import FP4_AMAX_FLOOR
from sglang.srt.runtime_context import get_platform

INDEX_HEAD_DIM = 128
# One index-K slot: 64 packed e2m1 bytes and four ue8m0 block exponents.
INDEX_K_PAYLOAD_BYTES = tl.constexpr(64)
INDEX_K_SCALE_BYTES = tl.constexpr(4)
INDEX_K_SLOT_BYTES = INDEX_K_PAYLOAD_BYTES.value + INDEX_K_SCALE_BYTES.value


@triton.jit
def _select_group_value(group, v0, v1, v2, v3):
    return tl.where(
        group == 0,
        v0,
        tl.where(group == 1, v1, tl.where(group == 2, v2, v3)),
    )


@triton.jit
def _ceil_ue8m0_exp(x):
    bits = x.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp += mantissa != 0
    return tl.minimum(tl.maximum(exp, 1), 254)


@triton.jit
def _fp4_e2m1_code(x):
    ax = tl.minimum(tl.abs(x), 6.0)
    idx = (ax > 0.25).to(tl.uint8)
    idx += (ax > 0.75).to(tl.uint8)
    idx += (ax > 1.25).to(tl.uint8)
    idx += (ax > 1.75).to(tl.uint8)
    idx += (ax > 2.5).to(tl.uint8)
    idx += (ax > 3.5).to(tl.uint8)
    idx += (ax > 5.0).to(tl.uint8)
    sign = ((x < 0) & (idx != 0)).to(tl.uint8)
    return idx | (sign << 3)


@triton.jit
def _fp4_e2m1_code_rne(x):
    """Round-to-nearest-even e2m1 code, matching the reference rounding."""
    ax = tl.minimum(tl.abs(x), 6.0)
    idx = (ax >= 0.25).to(tl.uint8)
    idx += (ax >= 0.75).to(tl.uint8)
    idx += (ax >= 1.25).to(tl.uint8)
    idx += (ax >= 1.75).to(tl.uint8)
    idx += (ax >= 2.5).to(tl.uint8)
    idx += (ax >= 3.5).to(tl.uint8)
    idx += (ax >= 5.0).to(tl.uint8)
    # Round-half-to-even: an odd index at an exact boundary drops to the even one.
    is_boundary = (
        (ax == 0.25)
        | (ax == 0.75)
        | (ax == 1.25)
        | (ax == 1.75)
        | (ax == 2.5)
        | (ax == 3.5)
        | (ax == 5.0)
    )
    idx = tl.where(is_boundary & ((idx & 1) == 1), idx - 1, idx)
    sign = ((x < 0) & (idx != 0)).to(tl.uint8)
    return idx | (sign << 3)


@triton.jit
def _quantize_fp4_indexer_kernel(
    x,
    x_fp4,
    x_sf,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    RNE: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    values = tl.load(x + token_id * BLOCK_N + offs).to(tl.float32)
    abs_values = tl.abs(values)

    amax0 = tl.max(tl.where(offs < GROUP_N, abs_values, 0.0), axis=0)
    amax1 = tl.max(
        tl.where((GROUP_N <= offs) & (offs < 2 * GROUP_N), abs_values, 0.0),
        axis=0,
    )
    amax2 = tl.max(
        tl.where((2 * GROUP_N <= offs) & (offs < 3 * GROUP_N), abs_values, 0.0),
        axis=0,
    )
    amax3 = tl.max(tl.where(3 * GROUP_N <= offs, abs_values, 0.0), axis=0)

    sf0 = tl.maximum(amax0 / 6.0, 1.0e-4)
    sf1 = tl.maximum(amax1 / 6.0, 1.0e-4)
    sf2 = tl.maximum(amax2 / 6.0, 1.0e-4)
    sf3 = tl.maximum(amax3 / 6.0, 1.0e-4)

    exp0 = _ceil_ue8m0_exp(sf0)
    exp1 = _ceil_ue8m0_exp(sf1)
    exp2 = _ceil_ue8m0_exp(sf2)
    exp3 = _ceil_ue8m0_exp(sf3)

    packed_sf = exp0 | (exp1 << 8) | (exp2 << 16) | (exp3 << 24)
    tl.store(x_sf + token_id, packed_sf)

    pair_offsets = tl.arange(0, BLOCK_N // 2)
    offs0 = pair_offsets * 2
    offs1 = offs0 + 1
    group0 = offs0 // GROUP_N
    group1 = offs1 // GROUP_N
    scale_exp0 = _select_group_value(group0, exp0, exp1, exp2, exp3)
    scale_exp1 = _select_group_value(group1, exp0, exp1, exp2, exp3)
    scale0 = (scale_exp0 << 23).to(tl.float32, bitcast=True)
    scale1 = (scale_exp1 << 23).to(tl.float32, bitcast=True)

    v0 = tl.load(x + token_id * BLOCK_N + offs0).to(tl.float32) / scale0
    v1 = tl.load(x + token_id * BLOCK_N + offs1).to(tl.float32) / scale1
    if RNE:
        code0 = _fp4_e2m1_code_rne(v0)
        code1 = _fp4_e2m1_code_rne(v1)
    else:
        code0 = _fp4_e2m1_code(v0)
        code1 = _fp4_e2m1_code(v1)
    packed = (code0 & 0x0F) | ((code1 & 0x0F) << 4)
    tl.store(x_fp4 + token_id * (BLOCK_N // 2) + pair_offsets, packed)


@triton.jit
def _quantize_fp4_indexer_rows(
    x,
    x_fp4,
    x_sf,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    RNE: tl.constexpr,
):
    tl.static_assert(BLOCK_N == 128 and GROUP_N == 32)
    # Each reduction covers one scale group. Keep its values for packing,
    # avoiding four masked full-row reductions and a second input load.
    group = tl.program_id(0) * BLOCK_M * 4 + tl.arange(0, BLOCK_M * 4)
    offs = tl.arange(0, GROUP_N)
    values = tl.load(
        x + group[:, None].to(tl.int64) * GROUP_N + offs[None, :],
        group[:, None] < M * 4,
        0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=1)
    exp = _ceil_ue8m0_exp(tl.maximum(amax / 6.0, 1.0e-4))
    scale = (exp << 23).to(tl.float32, bitcast=True)
    v0, v1 = tl.split(
        tl.reshape(values / scale[:, None], (BLOCK_M * 4, GROUP_N // 2, 2))
    )
    if RNE:
        code0 = _fp4_e2m1_code_rne(v0)
        code1 = _fp4_e2m1_code_rne(v1)
    else:
        code0 = _fp4_e2m1_code(v0)
        code1 = _fp4_e2m1_code(v1)
    packed = (code0 & 0x0F) | ((code1 & 0x0F) << 4)
    tl.store(
        x_fp4
        + group[:, None].to(tl.int64) * (GROUP_N // 2)
        + tl.arange(0, GROUP_N // 2)[None, :],
        packed,
        group[:, None] < M * 4,
    )
    # The four exponents occupy disjoint bytes, so integer sum packs them.
    shifts = tl.arange(0, 4) * 8
    packed_sf = tl.sum(
        tl.reshape(exp.to(tl.uint32), (BLOCK_M, 4)) << shifts[None, :], axis=1
    )
    token_id = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(x_sf + token_id, packed_sf.to(tl.int32), token_id < M)


@triton.jit
def _store_fp4_index_k_cache_kernel(
    k_fp4,
    k_sf,
    cache,
    loc,
    page_size: tl.constexpr,
    cache_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    cache_loc = tl.load(loc + token_id)
    page = cache_loc // page_size
    page_offset = cache_loc - page * page_size

    k = tl.load(k_fp4 + token_id * BLOCK + offsets)
    tl.store(cache + page * cache_stride + page_offset * BLOCK + offsets, k)

    sf = tl.load(k_sf + token_id)
    sf_offsets = tl.arange(0, 4)
    sf_bytes = (sf >> (sf_offsets * 8)) & 0xFF
    tl.store(
        cache + page * cache_stride + page_size * BLOCK + page_offset * 4 + sf_offsets,
        sf_bytes,
    )


def quantize_fp4_indexer_tensor(
    x: torch.Tensor, rne: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-32 ue8m0 fp4 quantize. rne=True uses round-to-nearest-even (the dsv41
    reference rounding); the default keeps ``_fp4_e2m1_code``'s thresholds."""
    assert x.shape[-1] == 128
    x = x.contiguous().view(-1, x.shape[-1])
    x_fp4 = torch.empty((x.shape[0], 64), device=x.device, dtype=torch.int8)
    x_sf = torch.empty((x.shape[0],), device=x.device, dtype=torch.int32)
    if x.shape[0] >= 4096 and get_platform().is_blackwell:
        # Independent rows share a CTA to avoid one block per 128 values.
        _quantize_fp4_indexer_rows[(triton.cdiv(x.shape[0], 8),)](
            x,
            x_fp4,
            x_sf,
            x.shape[0],
            8,
            BLOCK_N=128,
            GROUP_N=32,
            RNE=rne,
            num_warps=4,
        )
    elif x.shape[0] > 0:
        _quantize_fp4_indexer_kernel[(x.shape[0],)](
            x,
            x_fp4,
            x_sf,
            BLOCK_N=128,
            GROUP_N=32,
            RNE=rne,
        )
    return x_fp4, x_sf


def store_fp4_index_k_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    loc: torch.Tensor,
    *,
    page_size: int,
    rne: bool = False,
) -> None:
    assert input.shape[-1] == 128
    k_fp4, k_sf = quantize_fp4_indexer_tensor(input.contiguous(), rne=rne)
    n_tokens = input.numel() // input.shape[-1]
    assert k_fp4.shape == (n_tokens, 64)
    assert k_sf.shape == (n_tokens,)
    assert cache.shape[1] == page_size * INDEX_K_SLOT_BYTES

    if n_tokens == 0:
        return
    _store_fp4_index_k_cache_kernel[(n_tokens,)](
        k_fp4.view(torch.uint8),
        k_sf,
        cache,
        loc,
        page_size,
        cache.stride(0),
        BLOCK=64,
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
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(
        0, HALF_D
    )  # byte index i holds elements 2i (low nibble), 2i+1 (high nibble)

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
