# SPDX-License-Identifier: Apache-2.0
"""Depthwise FP32 conv1d / conv_transpose1d in the orders MIOpen's solvers use.

MIOpen picks a solver per shape for the alias-free resampling filters
(``[C, 1, K]`` weights, no padding or bias):

* conv1d on the im2col + GEMM solver: one FP32 FMA per tap, taps ascending
  (``conv1d_depthwise_fma``); the naive solver's order is ``conv1d_fp64acc``.
* conv_transpose1d, ``y[o] = sum x[(o - k) / stride] * w[k]`` over the taps
  with ``(o - k) % stride == 0``: the naive backward-data solver sums the
  products in FP64 with the taps ascending and rounds once (``fp64=True``);
  the GEMM + ``Col2Im2dU`` solver rounds each product to FP32 (one product per
  column entry) and col2im adds them in FP32 with the input positions
  ascending, i.e. the taps descending (``fp64=False``, ``taps_ascending=False``).

Products of FP32 values are exact in FP64; the FP32 orders keep their roundings
explicit (no FMA contraction beyond the one written).
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_BLOCK_T = 512


def _check(x: torch.Tensor, weight: torch.Tensor) -> int:
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError(f"expected FP32, got {x.dtype}/{weight.dtype}")
    channels = x.shape[1]
    if x.dim() != 3 or weight.shape[:2] != (channels, 1):
        raise ValueError(f"expected [B, {channels}, T] and [{channels}, 1, K]")
    return weight.shape[2]


@triton.jit
def _conv1d_depthwise_fma_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    channels,
    t_in,
    t_out,
    stride: tl.constexpr,
    taps: tl.constexpr,
    block_t: tl.constexpr,
):
    row = tl.program_id(0)  # batch * channels + channel
    channel = row % channels
    t = tl.program_id(1) * block_t + tl.arange(0, block_t)
    live = t < t_out
    x_row = x_ptr + row.to(tl.int64) * t_in
    acc = tl.zeros((block_t,), tl.float32)
    for tap in tl.static_range(taps):
        x = tl.load(x_row + t * stride + tap, mask=live, other=0.0)
        acc = tl.fma(x, tl.load(w_ptr + channel * taps + tap), acc)
    tl.store(out_ptr + row.to(tl.int64) * t_out + t, acc, mask=live)


def conv1d_depthwise_fma(
    x: torch.Tensor, weight: torch.Tensor, *, stride: int
) -> torch.Tensor:
    """``F.conv1d(x, weight, stride=stride, groups=C)``, FP32 FMA per tap."""
    taps = _check(x, weight)
    batch, channels, t_in = x.shape
    t_out = (t_in - taps) // stride + 1
    out = torch.empty(batch, channels, t_out, device=x.device, dtype=torch.float32)
    _conv1d_depthwise_fma_kernel[(batch * channels, triton.cdiv(t_out, _BLOCK_T))](
        x.contiguous(),
        weight.contiguous(),
        out,
        channels,
        t_in,
        t_out,
        stride=stride,
        taps=taps,
        block_t=_BLOCK_T,
    )
    return out


@triton.jit
def _conv_transpose1d_depthwise_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    channels,
    t_in,
    t_out,
    stride: tl.constexpr,
    taps: tl.constexpr,
    fp64: tl.constexpr,
    taps_ascending: tl.constexpr,
    block_t: tl.constexpr,
):
    row = tl.program_id(0)  # batch * channels + channel
    channel = row % channels
    o = tl.program_id(1) * block_t + tl.arange(0, block_t)
    live = o < t_out
    x_row = x_ptr + row.to(tl.int64) * t_in
    acc = tl.zeros((block_t,), tl.float64 if fp64 else tl.float32)
    for step in tl.static_range(taps):
        tap = step if taps_ascending else taps - 1 - step
        src = o - tap
        pos = src // stride
        ok = live & (src >= 0) & (src % stride == 0) & (pos < t_in)
        x = tl.load(x_row + pos, mask=ok, other=0.0)
        w = tl.load(w_ptr + channel * taps + tap)
        # a tap outside the input is skipped, not added as +0.0 (keeps a -0.0 sum)
        if fp64:
            acc = tl.where(ok, acc + x.to(tl.float64) * w.to(tl.float64), acc)
        else:
            acc = tl.where(ok, acc + x * w, acc)
    tl.store(out_ptr + row.to(tl.int64) * t_out + o, acc.to(tl.float32), mask=live)


def conv_transpose1d_depthwise(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    stride: int,
    fp64: bool,
    taps_ascending: bool = True,
) -> torch.Tensor:
    """``F.conv_transpose1d(x, weight, stride=stride, groups=C)`` in the order above."""
    taps = _check(x, weight)
    batch, channels, t_in = x.shape
    t_out = (t_in - 1) * stride + taps
    out = torch.empty(batch, channels, t_out, device=x.device, dtype=torch.float32)
    _conv_transpose1d_depthwise_kernel[
        (batch * channels, triton.cdiv(t_out, _BLOCK_T))
    ](
        x.contiguous(),
        weight.contiguous(),
        out,
        channels,
        t_in,
        t_out,
        stride=stride,
        taps=taps,
        fp64=fp64,
        taps_ascending=taps_ascending,
        block_t=_BLOCK_T,
        enable_fp_fusion=False,
    )
    return out
