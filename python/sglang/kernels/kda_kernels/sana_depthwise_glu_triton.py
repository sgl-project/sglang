# SPDX-License-Identifier: Apache-2.0
"""Channels-last BF16 depthwise convolution with rounded bias and GLU."""

import torch
import triton
import triton.language as tl

from sglang.kernels.numerics import round_bf16_to_fp32


@triton.jit
def _depthwise_glu_kernel(
    X,
    W,
    Bias,
    Out,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ht = tl.cdiv(H, 4)
    mask = offsets < N * ht * WIDTH * C
    channel = offsets % C
    pixel = offsets // C
    x = pixel % WIDTH
    y = (pixel // WIDTH) % ht * 4
    batch = pixel // (WIDTH * ht)
    hidden0 = tl.full((BLOCK,), 0, tl.float32)
    gate0 = tl.full((BLOCK,), 0, tl.float32)
    hidden1 = tl.full((BLOCK,), 0, tl.float32)
    gate1 = tl.full((BLOCK,), 0, tl.float32)
    hidden2 = tl.full((BLOCK,), 0, tl.float32)
    gate2 = tl.full((BLOCK,), 0, tl.float32)
    hidden3 = tl.full((BLOCK,), 0, tl.float32)
    gate3 = tl.full((BLOCK,), 0, tl.float32)
    for row in tl.static_range(6):
        for col in tl.static_range(3):
            sy = y + row - 1
            sx = x + col - 1
            valid = mask & (sy >= 0) & (sy < H) & (sx >= 0) & (sx < WIDTH)
            source = ((batch * H + sy) * WIDTH + sx) * (2 * C) + channel
            a = tl.load(X + source, valid, other=0).to(tl.float32)
            b = tl.load(X + source + C, valid, other=0).to(tl.float32)
            if row < 3:
                tap0 = row * 3 + col
                wa0 = tl.load(W + channel * 9 + tap0, mask, other=0).to(tl.float32)
                wb0 = tl.load(W + (channel + C) * 9 + tap0, mask, other=0).to(
                    tl.float32
                )
                hidden0 = tl.fma(a, wa0, hidden0)
                gate0 = tl.fma(b, wb0, gate0)
            if row >= 1 and row < 4:
                tap1 = (row - 1) * 3 + col
                wa1 = tl.load(W + channel * 9 + tap1, mask, other=0).to(tl.float32)
                wb1 = tl.load(W + (channel + C) * 9 + tap1, mask, other=0).to(
                    tl.float32
                )
                hidden1 = tl.fma(a, wa1, hidden1)
                gate1 = tl.fma(b, wb1, gate1)
            if row >= 2 and row < 5:
                tap2 = (row - 2) * 3 + col
                wa2 = tl.load(W + channel * 9 + tap2, mask, other=0).to(tl.float32)
                wb2 = tl.load(W + (channel + C) * 9 + tap2, mask, other=0).to(
                    tl.float32
                )
                hidden2 = tl.fma(a, wa2, hidden2)
                gate2 = tl.fma(b, wb2, gate2)
            if row >= 3 and row < 6:
                tap3 = (row - 3) * 3 + col
                wa3 = tl.load(W + channel * 9 + tap3, mask, other=0).to(tl.float32)
                wb3 = tl.load(W + (channel + C) * 9 + tap3, mask, other=0).to(
                    tl.float32
                )
                hidden3 = tl.fma(a, wa3, hidden3)
                gate3 = tl.fma(b, wb3, gate3)
    bh = tl.load(Bias + channel, mask, other=0).to(tl.float32)
    bg = tl.load(Bias + channel + C, mask, other=0).to(tl.float32)
    hidden0 = round_bf16_to_fp32(round_bf16_to_fp32(hidden0) + bh)
    gate0 = round_bf16_to_fp32(round_bf16_to_fp32(gate0) + bg)
    gate0 = round_bf16_to_fp32(gate0 * tl.sigmoid(gate0))
    dest0 = ((batch * H + y) * WIDTH + x) * C + channel
    tl.store(Out + dest0, hidden0 * gate0, mask & (y < H))
    hidden1 = round_bf16_to_fp32(round_bf16_to_fp32(hidden1) + bh)
    gate1 = round_bf16_to_fp32(round_bf16_to_fp32(gate1) + bg)
    gate1 = round_bf16_to_fp32(gate1 * tl.sigmoid(gate1))
    dest1 = ((batch * H + y + 1) * WIDTH + x) * C + channel
    tl.store(Out + dest1, hidden1 * gate1, mask & (y + 1 < H))
    hidden2 = round_bf16_to_fp32(round_bf16_to_fp32(hidden2) + bh)
    gate2 = round_bf16_to_fp32(round_bf16_to_fp32(gate2) + bg)
    gate2 = round_bf16_to_fp32(gate2 * tl.sigmoid(gate2))
    dest2 = ((batch * H + y + 2) * WIDTH + x) * C + channel
    tl.store(Out + dest2, hidden2 * gate2, mask & (y + 2 < H))
    hidden3 = round_bf16_to_fp32(round_bf16_to_fp32(hidden3) + bh)
    gate3 = round_bf16_to_fp32(round_bf16_to_fp32(gate3) + bg)
    gate3 = round_bf16_to_fp32(gate3 * tl.sigmoid(gate3))
    dest3 = ((batch * H + y + 3) * WIDTH + x) * C + channel
    tl.store(Out + dest3, hidden3 * gate3, mask & (y + 3 < H))


def fused_depthwise_conv3x3_bias_glu(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Stride-one, padding-one depthwise 3x3 followed by BF16 bias/SiLU/GLU.

    Each program reuses input rows for four adjacent output rows. The FP32
    accumulators visit taps in row-major order, then preserve the existing
    BF16 convolution, bias and SiLU materialization boundaries.
    """
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype is torch.bfloat16
        and x.ndim == 4
        and x.numel() > 0
        and x.shape[1] % 2 == 0
        and x.is_contiguous(memory_format=torch.channels_last)
        and weight.shape == (x.shape[1], 1, 3, 3)
        and weight.is_contiguous()
        and weight.dtype == x.dtype
        and weight.device == x.device
        and bias is not None
        and bias.shape == (x.shape[1],)
        and bias.is_contiguous()
        and bias.dtype == x.dtype
        and bias.device == x.device
    ):
        raise ValueError(
            "Expected channels-last BF16 CUDA input, depthwise 3x3 weight and channel bias"
        )
    n, dc, h, w = x.shape
    out = torch.empty(
        (n, dc // 2, h, w),
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last,
    )
    with torch.cuda.device(x.device):
        _depthwise_glu_kernel[
            (triton.cdiv(n * triton.cdiv(h, 4) * w * (dc // 2), 256),)
        ](
            x,
            weight,
            bias,
            out,
            n,
            dc // 2,
            h,
            w,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
