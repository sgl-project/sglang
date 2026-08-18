# SPDX-License-Identifier: Apache-2.0
"""Bit-exact post-processing kernels for Sana's channels-last GLUMB convs."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.triton.numerics import round_bf16_to_fp32


@triton.jit
def _bias_silu_kernel(out_ptr, x_ptr, bias_ptr, numel, channels: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024)
    mask = offsets < numel
    channel = offsets % channels
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
):
    offsets = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024)
    mask = offsets < out_numel
    channel = offsets % channels
    pixel = offsets // channels
    in_base = pixel * (2 * channels) + channel

    hidden = tl.load(x_ptr + in_base, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(x_ptr + in_base + channels, mask=mask, other=0.0).to(tl.float32)
    hidden_bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    gate_bias = tl.load(bias_ptr + channels + channel, mask=mask, other=0.0).to(
        tl.float32
    )

    hidden = round_bf16_to_fp32(hidden + hidden_bias)
    gate = round_bf16_to_fp32(gate + gate_bias)
    # SiLU materializes a bf16 tensor before the following multiply in eager.
    gate = round_bf16_to_fp32(gate * tl.sigmoid(gate))
    tl.store(out_ptr + offsets, hidden * gate, mask=mask)


def _is_channels_last_bf16(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and x.dim() == 4
        and x.numel() > 0
        and x.is_contiguous(memory_format=torch.channels_last)
    )


def can_use_fused_bias_silu(x: torch.Tensor, bias: torch.Tensor) -> bool:
    return (
        _is_channels_last_bf16(x)
        and bias.is_cuda
        and bias.dtype is x.dtype
        and bias.device == x.device
        and bias.dim() == 1
        and bias.shape[0] == x.shape[1]
        and bias.is_contiguous()
    )


def fused_bias_silu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if not can_use_fused_bias_silu(x, bias):
        raise RuntimeError("unsupported input for Sana fused bias-SiLU")
    out = torch.empty_like(x, memory_format=torch.preserve_format)
    with torch.cuda.device(x.device):
        _bias_silu_kernel[(triton.cdiv(x.numel(), 1024),)](
            out, x, bias, x.numel(), channels=x.shape[1]
        )
    return out


def can_use_fused_bias_glu(x: torch.Tensor, bias: torch.Tensor) -> bool:
    return (
        _is_channels_last_bf16(x)
        and x.shape[1] % 2 == 0
        and bias.is_cuda
        and bias.dtype is x.dtype
        and bias.device == x.device
        and bias.dim() == 1
        and bias.shape[0] == x.shape[1]
        and bias.is_contiguous()
    )


def fused_bias_glu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if not can_use_fused_bias_glu(x, bias):
        raise RuntimeError("unsupported input for Sana fused bias-GLU")
    batch, double_channels, height, width = x.shape
    channels = double_channels // 2
    out = torch.empty(
        (batch, channels, height, width),
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last,
    )
    with torch.cuda.device(x.device):
        _bias_glu_kernel[(triton.cdiv(out.numel(), 1024),)](
            out, x, bias, out.numel(), channels=channels
        )
    return out


__all__ = [
    "can_use_fused_bias_glu",
    "can_use_fused_bias_silu",
    "fused_bias_glu",
    "fused_bias_silu",
]
