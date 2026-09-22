"""Collapse the hyper-connection streams and normalize the sublayer input."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.layernorm.mxfp8_epilogue import mxfp8_epilogue


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
    """Fuse four-stream combine and RMSNorm for BF16 batches of width 5120."""
    m = x.shape[0]
    assert (0 < m <= 96 or 4096 <= m <= 65536) and x.shape == (m, 20480)
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
    parts = 4 if m <= 8 else (2 if m <= 48 else 1)
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
    mxfp8_epilogue(
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
