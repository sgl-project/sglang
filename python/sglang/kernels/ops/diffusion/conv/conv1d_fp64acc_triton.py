# SPDX-License-Identifier: Apache-2.0
"""FP32 conv1d accumulated in FP64, in MIOpen's naive-solver order.

MIOpen serves BigVGAN's long, narrow FP32 conv1d calls (8-256 channels, taps
3/7/11, dilation 1/3/5) with ``naive_conv_ab_nonpacked_fwd_nchw_float_double_float``:
one thread per output element sums ``x * w`` in FP64, input channel outer and
tap inner, skipping padded taps, then rounds to FP32. Products of FP32 values
are exact in FP64, so repeating the order repeats the sum. That kernel takes
4-5 ms per call; this one, tiled over output positions, takes tens of
microseconds. Bias stays with the caller, as in the eager path.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_BLOCK_T = 512


@triton.jit
def _conv1d_fp64acc_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    c_in,
    t_in,
    t_out,
    c_out,
    c_per_group,
    out_per_group,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    taps: tl.constexpr,
    block_t: tl.constexpr,
):
    row = tl.program_id(0)  # batch * c_out + output channel
    batch = row // c_out
    oc = row % c_out
    group = oc // out_per_group
    t = tl.program_id(1) * block_t + tl.arange(0, block_t)
    live = t < t_out
    acc = tl.zeros((block_t,), tl.float64)
    for ic in range(c_per_group):
        x_row = x_ptr + (batch * c_in + group * c_per_group + ic).to(tl.int64) * t_in
        w_row = w_ptr + (oc * c_per_group + ic) * taps
        for tap in tl.static_range(taps):
            pos = t * stride - padding + tap * dilation
            ok = live & (pos >= 0) & (pos < t_in)
            x = tl.load(x_row + pos, mask=ok, other=0.0)
            acc += x.to(tl.float64) * tl.load(w_row + tap).to(tl.float64)
    tl.store(out_ptr + row.to(tl.int64) * t_out + t, acc.to(tl.float32), mask=live)


def conv1d_fp64acc(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    """``F.conv1d(x, weight)`` (no bias) for FP32 ``[B, C, T]`` inputs."""
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError(f"conv1d_fp64acc takes FP32, got {x.dtype}/{weight.dtype}")
    if x.dim() != 3 or weight.dim() != 3:
        raise ValueError(
            f"expected [B, C, T] and [O, C/g, K], got {x.shape}/{weight.shape}"
        )
    batch, c_in, t_in = x.shape
    c_out, c_per_group, taps = weight.shape
    if c_in != c_per_group * groups or c_out % groups:
        raise ValueError(f"channels {c_in}->{c_out} do not split into {groups} groups")
    t_out = (t_in + 2 * padding - dilation * (taps - 1) - 1) // stride + 1
    out = torch.empty(batch, c_out, t_out, device=x.device, dtype=torch.float32)
    grid = (batch * c_out, triton.cdiv(t_out, _BLOCK_T))
    _conv1d_fp64acc_kernel[grid](
        x.contiguous(),
        weight.contiguous(),
        out,
        c_in,
        t_in,
        t_out,
        c_out,
        c_per_group,
        c_out // groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
        taps=taps,
        block_t=_BLOCK_T,
    )
    return out
