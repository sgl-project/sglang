# SPDX-License-Identifier: Apache-2.0
"""Temporal replication and spatial reflection padding in one pass."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pad(
    X,
    Y,
    N,
    shape: tl.constexpr,
    strides: tl.constexpr,
    padding: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _, C, T, H, W = shape
    SN, SC, ST, SH, SW = strides
    LEFT, RIGHT, PH, PW = padding
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    c = i % C
    p = i // C
    w = p % (W + 2 * PW) - PW
    p = p // (W + 2 * PW)
    h = p % (H + 2 * PH) - PH
    p = p // (H + 2 * PH)
    t = tl.minimum(tl.maximum(p % (T + LEFT + RIGHT) - LEFT, 0), T - 1)
    b = p // (T + LEFT + RIGHT)
    w = tl.minimum(tl.abs(w), 2 * (W - 1) - w)
    h = tl.minimum(tl.abs(h), 2 * (H - 1) - h)
    value = tl.load(X + b * SN + c * SC + t * ST + h * SH + w * SW, i < N, 0)
    tl.store(Y + i, value, i < N)


def replicate_reflect_pad3d(
    x: torch.Tensor, left: int, right: int, ph: int, pw: int
) -> torch.Tensor:
    """Pad NCDHW input, returning channels-last storage for a following conv3d."""
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.ndim == 5
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.shape[1] > 0
        and x.shape[2] > 0
        and min(left, right, ph, pw) >= 0
        and ph < x.shape[3]
        and pw < x.shape[4]
    ):
        raise ValueError(
            "replicate_reflect_pad3d expects CUDA NCDHW float input and valid padding"
        )
    b, c, t, h, w = x.shape
    y = torch.empty(
        (b, c, t + left + right, h + 2 * ph, w + 2 * pw),
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last_3d,
    )
    with torch.cuda.device(x.device):
        _pad[(triton.cdiv(y.numel(), 1024),)](
            x, y, y.numel(), x.shape, x.stride(), (left, right, ph, pw), 1024
        )
    return y
