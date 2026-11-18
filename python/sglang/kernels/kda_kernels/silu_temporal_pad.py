# SPDX-License-Identifier: Apache-2.0
"""SiLU, BTCHW-to-BCTHW layout, spatial reflection and causal padding."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _silu_temporal_pad(
    X,
    Y,
    N: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    col = offset % (W + 2) - 1
    row = offset // (W + 2) % (H + 2) - 1
    frame = offset // ((W + 2) * (H + 2)) % (T + 2) - 2
    channel = offset // ((W + 2) * (H + 2) * (T + 2)) % C
    batch = offset // ((W + 2) * (H + 2) * (T + 2) * C)
    col = tl.minimum(tl.abs(col), 2 * W - 2 - col)
    row = tl.minimum(tl.abs(row), 2 * H - 2 - row)
    source = (((batch * T + frame) * C + channel) * H + row) * W + col
    value = tl.load(X + source, mask=(offset < N) & (frame >= 0), other=0).to(
        tl.float32
    )
    # Match aten SiLU's opmath; approximate exp/div changes encoder outputs.
    activated = tl.div_rn(value, 1.0 + libdevice.exp(-value))
    tl.store(Y + offset, tl.where(frame >= 0, activated, 0.0), mask=offset < N)


def fused_silu_temporal_pad(x: torch.Tensor) -> torch.Tensor:
    """Activate contiguous BTCHW, reflect H/W by one, and zero-pad T by two.

    Returns contiguous BCTHW for a causal 3x3x3 convolution. SiLU retains
    the input dtype and aten's rounding before the padding/layout operations.
    """
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.ndim == 5
        and x.is_contiguous()
        and x.numel() > 0
        and min(x.shape[-2:]) > 1
    ):
        raise ValueError("expected contiguous CUDA FP16/BF16/FP32 BTCHW with H,W > 1")
    b, t, c, h, w = x.shape
    out = x.new_empty((b, c, t + 2, h + 2, w + 2))
    with torch.cuda.device(x.device):
        _silu_temporal_pad[(triton.cdiv(out.numel(), 256),)](
            x, out, out.numel(), c, t, h, w, 256, enable_fp_fusion=False
        )
    return out
