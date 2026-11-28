# SPDX-License-Identifier: Apache-2.0
"""Per-pixel RMS normalization and SiLU with channels-last 3D output."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _pixel_norm_silu(
    X,
    Y,
    S,
    shape: tl.constexpr,
    strides: tl.constexpr,
    EPS: tl.constexpr,
    BC: tl.constexpr,
    BP: tl.constexpr,
):
    _, C, D, H, W = shape
    SN, SC, SD, SH, SW = strides
    # A channel-by-position tile also coalesces NCDHW input reads.
    p = tl.program_id(0).to(tl.int64) * BP + tl.arange(0, BP)
    c = tl.arange(0, BC)
    offsets = (
        (p // (D * H * W)) * SN
        + ((p // (H * W)) % D) * SD
        + ((p // W) % H) * SH
        + (p % W) * SW
    )
    mask = (c[:, None] < C) & (p[None, :] < S)
    x = tl.load(X + c[:, None] * SC + offsets[None, :], mask, 0).to(tl.float32)
    dtype = Y.dtype.element_ty
    square = (x * x).to(dtype).to(tl.float32)
    variance = (tl.sum(square, 0) / C).to(dtype).to(tl.float32)
    denominator = (
        libdevice.sqrt_rn((variance + EPS).to(dtype).to(tl.float32))
        .to(dtype)
        .to(tl.float32)
    )
    normalized = tl.div_rn(x, denominator[None, :]).to(dtype).to(tl.float32)
    result = normalized / (1.0 + tl.exp(-normalized))
    tl.store(Y + p[None, :] * C + c[:, None], result, mask)


def pixel_norm_silu(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Fuse ``silu(x / sqrt(mean(x**2, dim=1) + eps))`` for NCDHW input.

    Keep the input-dtype rounding between operators. Channel reduction order
    may differ from torch, so callers must use the request's lossless gate.
    """
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.ndim == 5
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and 0 < x.shape[1] <= 2048
    ):
        raise ValueError(
            "pixel_norm_silu expects CUDA NCDHW float input with 1-2048 channels"
        )
    n, c, d, h, w = x.shape
    y = torch.empty_like(x, memory_format=torch.channels_last_3d)
    bc = triton.next_power_of_2(c)
    bp = max(1, 2048 // bc)
    if x.numel():
        with torch.cuda.device(x.device):
            _pixel_norm_silu[(triton.cdiv(n * d * h * w, bp),)](
                x,
                y,
                n * d * h * w,
                x.shape,
                x.stride(),
                eps,
                bc,
                bp,
                enable_fp_fusion=False,
            )
    return y
