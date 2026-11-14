# SPDX-License-Identifier: Apache-2.0
"""Fuse 2x resampling, SnakeBeta, and 2x downsampling for FP32 audio."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _alias_free_snake_beta(
    X,
    ALPHA,
    BETA,
    UP,
    DOWN,
    Y,
    T: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    UPSAMPLED_BLOCK: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    # The 12-tap filter needs 11 halo samples: 256 upsampled values serve
    # 123 outputs, avoiding an almost half-empty 512-element activation tile.
    outputs: tl.constexpr = (UPSAMPLED_BLOCK - 12) // 2 + 1
    start = tl.program_id(0).to(tl.int64) * outputs
    u = tl.arange(0, UPSAMPLED_BLOCK)
    # Replicate the cropped 2T signal before the downsampling filter.
    k = tl.minimum(tl.maximum(2 * start + u - 5, 0), 2 * T - 1)
    parity = (k + 15) % 2
    up = tl.full((UPSAMPLED_BLOCK,), 0, tl.float32)
    # A stride-2 transposed convolution uses six taps of matching parity.
    for j in tl.static_range(6):
        tap = parity + 2 * j
        sample = tl.minimum(tl.maximum((k + 5 - tap) // 2, 0), T - 1)
        up = up + tl.load(X + row * T + sample) * tl.load(UP + tap)
    up = up * 2.0
    # FP32 autocast promotes parameter exponentials as well as convolutions.
    alpha = libdevice.exp(tl.load(ALPHA + row % C).to(tl.float32))
    beta = libdevice.exp(tl.load(BETA + row % C).to(tl.float32))
    sine = libdevice.sin(up * alpha)
    activated = up + tl.div_rn(1.0, beta + EPS) * (sine * sine)
    o = tl.arange(0, BLOCK)
    result = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(12):
        result = result + tl.gather(
            activated, 2 * tl.minimum(o, outputs - 1) + j, axis=0
        ) * tl.load(DOWN + j)
    tl.store(Y + row * T + start + o, result, (o < outputs) & (start + o < T))


def alias_free_snake_beta(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    up_filter: torch.Tensor,
    down_filter: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Apply 12-tap upsample/SnakeBeta/downsample with log-scale channel parameters.

    Filter accumulation order can differ from cuDNN. Callers must use the
    request's lossless gate.
    """
    tensors = (x, alpha, beta, up_filter, down_filter)
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype == torch.float32
        and x.ndim == 3
        and x.numel() > 0
        and all(
            t.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and t.device == x.device
            and t.is_contiguous()
            for t in tensors
        )
        and alpha.shape == beta.shape == (x.shape[1],)
        and up_filter.numel() == down_filter.numel() == 12
    ):
        raise ValueError(
            "alias_free_snake_beta expects contiguous CUDA FP32 BCT input, float channel parameters and 12-tap filters"
        )
    y = torch.empty_like(x)
    block = 128
    upsampled_block = 2 * block
    outputs = (upsampled_block - 12) // 2 + 1
    with torch.cuda.device(x.device):
        _alias_free_snake_beta[
            (triton.cdiv(x.shape[-1], outputs), x.shape[0] * x.shape[1])
        ](
            x,
            alpha,
            beta,
            up_filter,
            down_filter,
            y,
            x.shape[-1],
            x.shape[1],
            eps,
            block,
            upsampled_block,
            enable_fp_fusion=False,
        )
    return y
