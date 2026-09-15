"""Fuse low-ratio RoPE, fake FP4 quantization and indexer packing/cache store.

Keep both quantization stages: the indexer packer has a different scale floor
from fake_quant_fp4, so directly packing the first stage is not equivalent.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    _ceil_ue8m0_exp,
    _fp4_e2m1_code_rne,
)
from sglang.kernels.ops.attention.dsv4.rope_fake_quant_fp4 import FP4_AMAX_FLOOR


@triton.jit
def _rope_fake_quant_pack_indexer_kernel(
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
    # Preserve the BF16 intermediate before recomputing the indexer scale,
    # including its 1e-4 lower bound.
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


def rope_fake_quant_pack_indexer(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rope_dim: int,
    *,
    positions: torch.Tensor | None = None,
    cache: torch.Tensor | None = None,
    loc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return packed [T*heads,64] / [T*heads], or write paged key cache.

    Without positions, freqs is already gathered per token; otherwise the
    kernel reads freqs[positions] directly, removing the gather launch.
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
        assert cache.ndim == 2 and cache.shape[1] % 68 == 0
        payload = scale = None
        page_size, cache_stride = cache.shape[1] // 68, cache.stride(0)
    if rows:
        _rope_fake_quant_pack_indexer_kernel[(rows,)](
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
