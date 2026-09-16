"""MXFP8 quantization written as an epilogue of the kernel that produces the row.

At the speculative BS=1 shapes (<=8 rows) the standalone FlashInfer
``mxfp8_quantize`` launch costs about as much as the norm it follows, even
though it only re-reads what that norm just wrote.  These kernels do the norm
and the quantization in one pass; the quantized values come from the same BF16
rounding the standalone pair produces, so both outputs are bitwise identical to
``rmsnorm`` followed by ``mxfp8_quantize(..., is_sf_swizzled_layout=True)``.

The scale factors use FlashInfer's 128x4 swizzle:
``off = (g // 4) * 512 + ((r % 32) * 4 + ((r // 32) % 4)) * 4 + (g % 4)``
over a row count padded to a multiple of 128, with the UE8M0 conversion
(positive rounding, subnormals included) that ``mxfp8_quantize`` uses.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# FlashInfer's positive-rounding UE8M0 conversion of a per-group amax, subnormals
# included: the scale byte and the multiplier that maps the group into e4m3 range.
@triton.jit
def ue8m0_scale(amax):
    normalized = amax * (1.0 / 448.0)
    bits = normalized.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.where(normalized <= 0, 0, tl.minimum(exponent + bump.to(tl.int32), 254))
    inv = tl.where(sf == 0, 0, ((254 - sf) << 23)).to(tl.float32, bitcast=True)
    return sf, inv


@triton.jit
def _mxfp8_epilogue(
    y, row, Q, S, K: tl.constexpr, BLOCK: tl.constexpr, GROUPS: tl.constexpr, g_lo, g_hi
):
    # Stores only groups in [g_lo, g_hi) so a row can be split across CTAs.
    GP: tl.constexpr = BLOCK // 32
    g = tl.arange(0, GP)
    gmask = (g < GROUPS) & (g >= g_lo) & (g < g_hi)
    e = tl.arange(0, 32)
    idx = g[:, None] * 32 + e[None, :]
    v = tl.reshape(y.to(tl.float32), (GP, 32))
    amax = tl.max(tl.abs(v), 1)
    sf, scale = ue8m0_scale(amax)
    q = tl.minimum(tl.maximum(v * scale[:, None], -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * K + idx, q, gmask[:, None])
    off = (g // 4) * 512 + ((row % 32) * 4 + ((row // 32) % 4)) * 4 + (g % 4)
    tl.store(S + off, sf.to(tl.uint8), gmask)


@triton.jit
def _hc_combine_norm_mxfp8_kernel(
    X,
    P,
    W,
    Y,
    Q,
    S,
    SX: tl.constexpr,
    SP: tl.constexpr,
    EPS: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
    SLICE: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    h = tl.arange(0, BLOCK)
    m = h < K
    value = tl.full((BLOCK,), 0, tl.float32)
    for c in tl.static_range(4):
        pre = tl.load(P + row * SP + c).to(tl.float32)
        x = tl.load(X + row * SX + c * K + h, m, 0).to(tl.float32)
        value += x * pre
    # The unfused combine stores BF16 before RMSNorm reads it.
    value = value.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / K + EPS)
    weight = tl.load(W + h, m, 0).to(tl.float32)
    y = (value * inv_rms * weight).to(tl.bfloat16)
    tl.store(Y + row * K + h, y, m & (h >= part * SLICE) & (h < (part + 1) * SLICE))
    _mxfp8_epilogue(
        y, row, Q, S, K, BLOCK, GROUPS, part * (SLICE // 32), (part + 1) * (SLICE // 32)
    )


def _parts_for(k: int) -> int:
    # Row splits: recomputing the statistic beats running 6 CTAs on 148 SMs.
    parts = 4
    while parts > 1 and (k % (parts * 32)):
        parts //= 2
    return parts


def _alloc(m, k, device):
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=device)
    s = torch.zeros(
        (k // 32) * (triton.cdiv(m, 128) * 128), dtype=torch.uint8, device=device
    )
    return q, s


def hc_combine_norm_mxfp8(
    x: torch.Tensor, pre: torch.Tensor, weight: torch.Tensor, eps: float
):
    """Four-stream combine + RMSNorm returning ``(y_bf16, y_q, y_sf)``."""
    m = x.shape[0]
    assert 0 < m <= 8, "the fused MXFP8 epilogue only supports small decode/verify"
    k = x.shape[1] // 4
    y = torch.empty((m, k), dtype=x.dtype, device=x.device)
    q, s = _alloc(m, k, x.device)
    parts = _parts_for(k)
    _hc_combine_norm_mxfp8_kernel[(m, parts)](
        x,
        pre,
        weight,
        y,
        q,
        s,
        SX=x.stride(0),
        SP=pre.stride(0),
        EPS=eps,
        K=k,
        BLOCK=triton.next_power_of_2(k),
        GROUPS=k // 32,
        SLICE=k // parts,
        num_warps=8,
    )
    return y, q, s


@triton.jit
def _rmsnorm_mxfp8_kernel(
    X,
    W,
    Y,
    Q,
    S,
    SX: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    if row < M:
        h = tl.arange(0, BLOCK)
        v = tl.load(X + row * SX + h, h < K, 0).to(tl.float32)
        weight = tl.load(W + h, h < K, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(v * v, 0) / K + EPS)
        y = (v * inv * weight).to(tl.bfloat16)
        tl.store(Y + row * K + h, y, h < K)
        _mxfp8_epilogue(y, row, Q, S, K, BLOCK, GROUPS, 0, GROUPS)
    else:
        # Padding scale entries are disjoint from the live rows above.
        off = (row - M) * 512 + tl.arange(0, 512)
        pad_row = ((off // 4) % 4) * 32 + ((off // 16) % 32)
        tl.store(S + off, 0, (off < GROUPS * 128) & (pad_row >= M))


def rmsnorm_mxfp8(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm returning ``(y_bf16, y_q, y_sf)`` with the same MXFP8 epilogue."""
    m, k = x.shape
    assert 0 < m <= 8 and k % 32 == 0
    assert x.dtype == weight.dtype == torch.bfloat16 and x.stride(1) == 1
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((k // 32) * 128, dtype=torch.uint8, device=x.device)
    _rmsnorm_mxfp8_kernel[(m + triton.cdiv(s.numel(), 512),)](
        x,
        weight,
        y,
        q,
        s,
        SX=x.stride(0),
        M=m,
        K=k,
        EPS=eps,
        BLOCK=triton.next_power_of_2(k),
        GROUPS=k // 32,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return y, q, s
