"""Collapse the hyper-connection streams and normalize the sublayer input."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_epilogue import mxfp8_epilogue


@triton.jit
def _hc_combine_norm(
    X,
    P,
    W,
    Y,
    SX: tl.constexpr,
    SP: tl.constexpr,
    EPS: tl.constexpr,
    PARTS: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    h = tl.arange(0, 8192)
    value = tl.full((8192,), 0, tl.float32)
    for c in tl.static_range(4):
        pre = tl.load(P + row * SP + c).to(tl.float32)
        x = tl.load(X + row * SX + c * 5120 + h, h < 5120, 0).to(tl.float32)
        value += x * pre
    # The unfused combine stores BF16 before RMSNorm reads it.
    value = value.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / 5120 + EPS)
    mask = (h >= part * (5120 // PARTS)) & (h < (part + 1) * (5120 // PARTS))
    weight = tl.load(W + h, mask, 0).to(tl.float32)
    tl.store(Y + row * 5120 + h, value * inv_rms * weight, mask)


@triton.jit
def _hc_combine_norm_prefill(
    X, P, W, Y, SX: tl.constexpr, SP: tl.constexpr, EPS: tl.constexpr
):
    # Large batches have enough rows to use one CTA per row without repeating
    # the combine and RMS reduction for each output partition.
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, 8192)
    value = tl.full((8192,), 0, tl.float32)
    for c in tl.static_range(4):
        pre = tl.load(P + row * SP + c).to(tl.float32)
        x = tl.load(X + row * SX + c * 5120 + h, h < 5120, 0).to(tl.float32)
        value += x * pre
    value = value.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / 5120 + EPS)
    weight = tl.load(W + h, h < 5120, 0).to(tl.float32)
    tl.store(Y + row * 5120 + h, value * inv_rms * weight, h < 5120)


def hc_combine_norm(
    x: torch.Tensor, pre: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Fuse four-stream combine and RMSNorm for BF16 batches of width 5120.

    Every row count takes the same per-row reduction in the same order, so the
    result is batch-invariant across the grid-split variants below."""
    m = x.shape[0]
    assert x.shape == (m, 20480)
    assert pre.shape == (m, 4) and pre.stride(1) == 1
    assert weight.shape == (5120,) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16 and x.stride(1) == 1
    y = torch.empty((m, 5120), dtype=x.dtype, device=x.device)
    if m >= 4096:
        _hc_combine_norm_prefill[(m,)](
            x, pre, weight, y, x.stride(0), pre.stride(0), eps, num_warps=4
        )
        return y
    # Four CTAs per row trade redundant statistics for more concurrent loads
    # when only a few speculative tokens are being processed; wider batches have
    # enough rows to split the 5120 columns fewer ways.
    parts = _parts_for(m, 5120)
    _hc_combine_norm[(m, parts)](
        x, pre, weight, y, x.stride(0), pre.stride(0), eps, parts, num_warps=8
    )
    return y


@triton.jit
def _hc_combine_norm_mxfp8_kernel(
    X,
    P,
    W,
    Y,
    Q,
    S,
    M,
    SX: tl.constexpr,
    SP: tl.constexpr,
    EPS: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
    SLICE: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    if row < M:
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
        slice_mask = (h >= part * SLICE) & (h < (part + 1) * SLICE)
        tl.store(Y + row * K + h, y, m & slice_mask)
        g_lo, g_hi = part * (SLICE // 32), (part + 1) * (SLICE // 32)
        mxfp8_epilogue(y, row, Q, S, K, BLOCK, GROUPS, g_lo, g_hi)
    else:
        # Padding scale entries are disjoint from the live rows above.
        off = ((row - M) * tl.num_programs(1) + part) * 512 + tl.arange(0, 512)
        pad_row = ((off // 4) % 4) * 32 + ((off // 16) % 32)
        tl.store(S + off, 0, (off < GROUPS * 128) & (pad_row >= M))


def _parts_for(m: int, k: int) -> int:
    parts = 4 if m <= 8 else (2 if m <= 48 else 1)
    while parts > 1 and (k % (parts * 32)):
        parts //= 2
    return parts


def _alloc(m, k, device):
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=device)
    # The kernel zeroes the padding rows' scales; no separate memset.
    s = torch.empty(
        (k // 32) * (triton.cdiv(m, 128) * 128), dtype=torch.uint8, device=device
    )
    return q, s


def hc_combine_norm_mxfp8(
    x: torch.Tensor, pre: torch.Tensor, weight: torch.Tensor, eps: float
):
    """Four-stream combine + RMSNorm returning ``(y_bf16, y_q, y_sf)``.

    Every CTA of a row computes the whole row's reduction in the same order, so
    the outputs do not depend on the grid split."""
    m = x.shape[0]
    assert m <= 128, "the MXFP8 scale swizzle addresses a single 128-row tile"
    k = x.shape[1] // 4
    y = torch.empty((m, k), dtype=x.dtype, device=x.device)
    q, s = _alloc(m, k, x.device)
    parts = _parts_for(m, k)
    # Programs past the last row each zero one 512-byte block of padding scales.
    pad_blocks = triton.cdiv(s.numel(), 512) if m % 128 else 0
    _hc_combine_norm_mxfp8_kernel[(m + triton.cdiv(pad_blocks, parts), parts)](
        x,
        pre,
        weight,
        y,
        q,
        s,
        m,
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
