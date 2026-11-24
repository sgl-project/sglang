# SPDX-License-Identifier: Apache-2.0
"""Triton versions of the channels-last VAE norm kernels, for platforms without the CUDA JIT kernels.

- ``channel_rmsnorm_silu_nhwc_triton``: the contract of ``channel_rmsnorm_silu_nhwc``,
  ``silu(bf16(bf16(bf16(bf16(x' / max(||x'||, 1e-12)) * scale) * gamma) + 0))`` per pixel
  with ``x' = bf16(x + bias)``. The rounding points match the eager chain; only the
  channel reduction order differs, so callers keep it behind the quality gate.
- ``bias_residual_add_triton``: ``bf16(bf16(y + bias) + h)``, bit-exact against aten's
  two adds.

bf16 only; channels-last dense tensors (and NCHW-contiguous ones for the residual add).
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.norm.channel_rmsnorm_silu_nhwc_jit import (
    is_channels_last_dense,
)

_MAX_CHANNELS = 2048


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ELEMENTS": e}, num_warps=w)
        for e in (2048, 4096, 8192, 16384)
        for w in (4, 8)
    ],
    key=["channels"],
)
@triton.jit
def _channel_rmsnorm_silu_kernel(
    out_ptr,
    x_ptr,
    gamma_ptr,
    bias_ptr,
    rows,
    channels,
    scale,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
):
    # Every config has BLOCK_ELEMENTS >= _MAX_CHANNELS >= BLOCK_C.
    BLOCK_ROWS: tl.constexpr = BLOCK_ELEMENTS // BLOCK_C
    row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, BLOCK_C)
    col_mask = col < channels
    mask = (row[:, None] < rows) & col_mask[None, :]
    offsets = row[:, None] * channels + col[None, :]
    value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + col, mask=col_mask, other=0.0).to(tl.float32)
        value = (value + bias[None, :]).to(tl.bfloat16).to(tl.float32)
        value = tl.where(mask, value, 0.0)
    norm = tl.sqrt_rn(tl.sum(value * value, axis=1))
    denominator = tl.maximum(norm, 1e-12)
    value = tl.math.div_rn(value, denominator[:, None]).to(tl.bfloat16).to(tl.float32)
    value = (value * scale).to(tl.bfloat16).to(tl.float32)
    gamma = tl.load(gamma_ptr + col, mask=col_mask, other=0.0).to(tl.float32)
    value = (value * gamma[None, :]).to(tl.bfloat16).to(tl.float32)
    value = tl.math.div_rn(value, 1.0 + tl.exp(-value))
    tl.store(out_ptr + offsets, value.to(tl.bfloat16), mask=mask)


@triton.jit
def _bias_residual_add_kernel(
    out_ptr,
    y_ptr,
    h_ptr,
    bias_ptr,
    numel,
    channels,
    spatial,
    CHANNEL_INNER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    if CHANNEL_INNER:
        channel = offsets % channels
    else:
        channel = (offsets // spatial) % channels
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(h_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    biased = (y + bias).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + offsets, (biased + h).to(tl.bfloat16), mask=mask)


def _channel_last_dims(x: torch.Tensor) -> tuple[int, ...]:
    return (0, *range(2, x.ndim), 1)


def _is_channel_vector(t: torch.Tensor, x: torch.Tensor) -> bool:
    return (
        t.device == x.device
        and t.dtype is torch.bfloat16
        and t.numel() == x.shape[1]
        and t.is_contiguous()
    )


def can_use_channel_rmsnorm_silu_nhwc_triton(
    x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    return (
        x.is_cuda
        and not torch.compiler.is_compiling()
        and x.dtype is torch.bfloat16
        and x.numel() > 0
        and is_channels_last_dense(x)
        and x.shape[1] <= _MAX_CHANNELS
        and _is_channel_vector(gamma, x)
        and (bias is None or _is_channel_vector(bias, x))
    )


def channel_rmsnorm_silu_nhwc_triton(
    x: torch.Tensor,
    gamma: torch.Tensor,
    scale: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``silu(rmsnorm(x + bias))`` over the channel dim of a channels-last tensor; output keeps ``x``'s strides."""
    out = torch.empty_like(x)
    channels = x.shape[1]
    rows = x.numel() // channels
    block_c = triton.next_power_of_2(channels)
    dims = _channel_last_dims(x)

    def grid(meta):
        return (triton.cdiv(rows, max(1, meta["BLOCK_ELEMENTS"] // block_c)),)

    with torch.cuda.device(x.device):
        _channel_rmsnorm_silu_kernel[grid](
            out.permute(*dims),
            x.permute(*dims),
            gamma,
            gamma if bias is None else bias,
            rows,
            channels,
            float(scale),
            HAS_BIAS=bias is not None,
            BLOCK_C=block_c,
        )
    return out


def can_use_bias_residual_add_triton(
    y: torch.Tensor, bias: torch.Tensor, h: torch.Tensor
) -> bool:
    return (
        y.is_cuda
        and not torch.compiler.is_compiling()
        and y.ndim in (4, 5)
        and y.numel() > 0
        and y.dtype is torch.bfloat16
        and h.dtype is torch.bfloat16
        and bias.dtype is torch.bfloat16
        and h.device == y.device
        and bias.device == y.device
        and h.shape == y.shape
        and h.stride() == y.stride()
        and bias.numel() == y.shape[1]
        and bias.is_contiguous()
        and (is_channels_last_dense(y) or y.is_contiguous())
    )


def bias_residual_add_triton(
    y: torch.Tensor, bias: torch.Tensor, h: torch.Tensor
) -> torch.Tensor:
    """``(y + bias) + h`` with aten's per-op bf16 rounding; output keeps ``y``'s strides."""
    out = torch.empty_like(y)
    channels = y.shape[1]
    inner = is_channels_last_dense(y) and not y.is_contiguous()
    spatial = y.numel() // (y.shape[0] * channels)
    block = 2048
    with torch.cuda.device(y.device):
        _bias_residual_add_kernel[(triton.cdiv(y.numel(), block),)](
            out,
            y,
            h,
            bias,
            y.numel(),
            channels,
            spatial,
            CHANNEL_INNER=inner,
            BLOCK=block,
            num_warps=4,
        )
    return out
