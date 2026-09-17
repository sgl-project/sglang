from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.attention.dsv4.torch_quant import FP4_AMAX_FLOOR

# One index-K slot: 64 packed e2m1 bytes and four ue8m0 block exponents.
INDEX_K_SLOT_BYTES = 64 + 4


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
    if x.shape[0] > 0:
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
