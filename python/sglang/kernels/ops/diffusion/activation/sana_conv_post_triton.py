# SPDX-License-Identifier: Apache-2.0
"""Bit-exact post-processing kernels for Sana's GLUMB convs."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.common.numerics import round_bf16_to_fp32


@triton.jit
def _bias_silu_kernel(
    out_ptr,
    x_ptr,
    bias_ptr,
    numel,
    channels: tl.constexpr,
    spatial: tl.constexpr,
    channels_last: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024)
    mask = offsets < numel
    if channels_last:
        channel = offsets % channels
    else:
        channel = (offsets // spatial) % channels
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    # nn.Conv2d applies its bf16 bias before nn.SiLU, so preserve the
    # intermediate bf16 rounding boundary rather than contracting the chain.
    biased = round_bf16_to_fp32(x + bias)
    tl.store(out_ptr + offsets, biased * tl.sigmoid(biased), mask=mask)


@triton.jit
def _bias_glu_kernel(
    out_ptr,
    x_ptr,
    bias_ptr,
    out_numel,
    channels: tl.constexpr,
    spatial: tl.constexpr,
    channels_last: tl.constexpr,
    has_bias: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024)
    mask = offsets < out_numel
    if channels_last:
        channel = offsets % channels
        pixel = offsets // channels
        in_base = pixel * (2 * channels) + channel
        gate_offset = channels
    else:
        channel = (offsets // spatial) % channels
        batch = offsets // (channels * spatial)
        in_base = batch * (2 * channels * spatial) + offsets % (channels * spatial)
        gate_offset = channels * spatial

    hidden = tl.load(x_ptr + in_base, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(x_ptr + in_base + gate_offset, mask=mask, other=0.0).to(tl.float32)
    if has_bias:
        hidden_bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        gate_bias = tl.load(bias_ptr + channels + channel, mask=mask, other=0.0).to(
            tl.float32
        )
        hidden = round_bf16_to_fp32(hidden + hidden_bias)
        gate = round_bf16_to_fp32(gate + gate_bias)
    # SiLU materializes a bf16 tensor before the following multiply in eager.
    gate = round_bf16_to_fp32(gate * tl.sigmoid(gate))
    tl.store(out_ptr + offsets, hidden * gate, mask=mask)


def _validate_conv_post(x: torch.Tensor, bias: torch.Tensor | None) -> None:
    if not (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and x.ndim == 4
        and x.numel() > 0
        and (x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last))
    ):
        raise RuntimeError(
            "Sana conv post-processing expects a dense BF16 CUDA NCHW tensor"
        )
    if bias is not None and not (
        bias.dtype == x.dtype
        and bias.device == x.device
        and bias.shape == (x.shape[1],)
        and bias.is_contiguous()
    ):
        raise RuntimeError(
            "bias must be a contiguous channel vector matching x's dtype and device"
        )


def fused_bias_silu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    _validate_conv_post(x, bias)
    out = torch.empty_like(x, memory_format=torch.preserve_format)
    with torch.cuda.device(x.device):
        _bias_silu_kernel[(triton.cdiv(x.numel(), 1024),)](
            out,
            x,
            bias,
            x.numel(),
            channels=x.shape[1],
            spatial=x.shape[2] * x.shape[3],
            channels_last=x.is_contiguous(memory_format=torch.channels_last),
        )
    return out


def fused_bias_glu(x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Apply optional bias then ``hidden * silu(gate)`` along the channel axis.

    Pass no bias for an already biased native depthwise-convolution output:
    splitting that convolution's bias can change its accumulation rounding.
    """
    _validate_conv_post(x, bias)
    if x.shape[1] % 2:
        raise RuntimeError("Sana GLU requires an even channel count")
    batch, double_channels, height, width = x.shape
    channels = double_channels // 2
    channels_last = x.is_contiguous(memory_format=torch.channels_last)
    out = torch.empty(
        (batch, channels, height, width),
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last if channels_last else torch.contiguous_format,
    )
    with torch.cuda.device(x.device):
        _bias_glu_kernel[(triton.cdiv(out.numel(), 1024),)](
            out,
            x,
            bias,
            out.numel(),
            channels=channels,
            spatial=height * width,
            channels_last=channels_last,
            has_bias=bias is not None,
        )
    return out


__all__ = [
    "fused_bias_glu",
    "fused_bias_silu",
]
