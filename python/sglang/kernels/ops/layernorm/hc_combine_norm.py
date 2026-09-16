"""Collapse the hyper-connection streams and normalize the sublayer input."""

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_combine_norm(X, P, W, Y, SX: tl.constexpr, SP: tl.constexpr, EPS: tl.constexpr):
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
    mask = (h >= part * 1280) & (h < (part + 1) * 1280)
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
    assert (0 < m <= 8 or 4096 <= m <= 65536) and x.shape == (m, 20480)
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
    # when only a few speculative tokens are being processed.
    _hc_combine_norm[(m, 4)](
        x, pre, weight, y, x.stride(0), pre.stride(0), eps, num_warps=8
    )
    return y
