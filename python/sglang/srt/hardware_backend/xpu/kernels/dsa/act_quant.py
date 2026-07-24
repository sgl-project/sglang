"""Torch-native per-group FP8 activation quantization for XPU."""

from typing import Optional, Tuple

import torch


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group FP8 activation quantization using PyTorch ops.

    For each group of `block_size` columns, computes the abs-max, derives a
    per-group scale, and quantizes to float8_e4m3fn.

    Args:
        x: Input tensor (contiguous, last dim divisible by block_size).
        block_size: Number of columns per quantization group.
        scale_fmt: If not None, round scales to nearest power of 2.

    Returns:
        (y_fp8, scale) where y_fp8 has dtype float8_e4m3fn and scale is float32.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    N = x.size(-1)
    assert (
        N % block_size == 0
    ), f"Last dim must be divisible by block_size ({block_size})"

    FP8_MAX = 448.0
    x_flat = x.view(-1, N).float()
    M = x_flat.size(0)
    n_groups = N // block_size

    # Reshape to (M, n_groups, block_size) for per-group quantization
    x_grouped = x_flat.view(M, n_groups, block_size)
    amax = x_grouped.abs().amax(dim=-1).clamp(min=1e-4)  # (M, n_groups)

    if scale_fmt is not None:
        # Round scale to power of 2
        scale = torch.exp2(torch.log2(amax / FP8_MAX).ceil())
    else:
        scale = amax / FP8_MAX

    # Quantize
    y = (x_grouped / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    y = y.view(M, N).to(torch.float8_e4m3fn).view(*x.shape[:-1], N)
    scale = scale.view(*x.shape[:-1], n_groups)
    return y, scale
