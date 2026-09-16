"""Collapse the hyper-connection streams and normalize the sublayer input."""

import os

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


def hc_combine_norm(
    x: torch.Tensor, pre: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Fuse four-stream combine and RMSNorm for small BF16 batches of width 5120."""
    m = x.shape[0]
    assert 0 < m <= 8 and x.shape == (m, 20480)
    assert pre.shape == (m, 4) and pre.stride(1) == 1
    assert weight.shape == (5120,) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16 and x.stride(1) == 1
    y = torch.empty((m, 5120), dtype=x.dtype, device=x.device)
    # Four CTAs per row trade redundant statistics for more concurrent loads
    # when only a few speculative tokens are being processed.
    _hc_combine_norm[(m, 4)](
        x, pre, weight, y, x.stride(0), pre.stride(0), eps, num_warps=8
    )
    return y


# --- prefill variant -------------------------------------------------------
# The kernel above splits every row across four CTAs, which trades a 4x
# re-read of the row for extra concurrency when only a handful of speculative
# tokens are in flight. At the prefill shape (thousands of rows) concurrency is
# already saturated and that re-read is pure cost, so the prefill variant runs
# one CTA per row. It removes the [m, 5120] BF16 intermediate that the unfused
# combine + RMSNorm pair round-trips through memory, and reproduces that pair's
# BF16 store of the combine result so the result tracks it.
MHC_FUSE_PREFILL_NORM = os.environ.get("SGLANG_MHC_FUSE_PREFILL_NORM", "0") == "1"


@triton.jit
def _hc_combine_norm_prefill(
    X, P, W, Y, SX, SP, SY, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK)
    mask = h < 5120
    value = tl.zeros((BLOCK,), tl.float32)
    for c in tl.static_range(4):
        pre = tl.load(P + row * SP + c).to(tl.float32)
        x = tl.load(X + row * SX + c * 5120 + h, mask, 0.0).to(tl.float32)
        value += x * pre
    # The unfused combine stores BF16 before RMSNorm reads it.
    value = value.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / 5120 + EPS)
    weight = tl.load(W + h, mask, 0.0).to(tl.float32)
    tl.store(Y + row * SY + h, value * inv_rms * weight, mask)


def hc_combine_norm_prefill(
    x: torch.Tensor, pre: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Fuse four-stream combine and RMSNorm at the prefill shape, width 5120."""
    m = x.shape[0]
    assert m > 0 and x.shape == (m, 20480)
    assert pre.shape == (m, 4) and pre.stride(1) == 1
    assert weight.shape == (5120,) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16 and x.stride(1) == 1
    y = torch.empty((m, 5120), dtype=x.dtype, device=x.device)
    _hc_combine_norm_prefill[(m,)](
        x, pre, weight, y, x.stride(0), pre.stride(0), y.stride(0),
        EPS=eps, BLOCK=8192, num_warps=4, num_stages=2,
    )
    return y
