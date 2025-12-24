"""Quantization utilities for TileLang GEMM."""

from typing import Tuple

import torch

from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8


def per_token_cast_to_fp8(
    x: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group FP8 quantization for activation.

    Returns (x_fp8, x_scale) where x_scale is row-major.
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    assert (
        x.shape[-1] % group_size == 0
    ), f"K={x.shape[-1]} must be divisible by group_size={group_size}"

    x_fp8, x_scale = sglang_per_token_group_quant_fp8(
        x.contiguous(),
        group_size=group_size,
        column_major_scales=False,
    )

    return x_fp8, x_scale


def per_block_cast_to_fp8_weight(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-block FP8 quantization for weight.

    Returns (x_fp8, x_scale) where x_scale shape is (N//128, K//128).
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    return per_block_cast_to_fp8(x.contiguous())


def prepare_gemm_inputs(
    A: torch.Tensor,
    B: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare FP8 inputs for TileLang GEMM.

    Returns (A_fp8, B_fp8, A_scale, B_scale).
    """
    A_fp8, A_scale = per_token_cast_to_fp8(A, group_size)
    B_fp8, B_scale = per_block_cast_to_fp8_weight(B)
    return A_fp8, B_fp8, A_scale, B_scale


__all__ = [
    "per_token_cast_to_fp8",
    "per_block_cast_to_fp8_weight",
    "prepare_gemm_inputs",
]
