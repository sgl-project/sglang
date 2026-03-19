# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for ModelOpt NVFP4 weight/activation layout."""

from __future__ import annotations

import torch

# FP4 GEMM alignment constant - CUTLASS/FlashInfer kernels require dimensions divisible by 32
FP4_GEMM_ALIGNMENT = 32


def round_up_to_multiple(x: int, m: int) -> int:
    """Round up x to the nearest multiple of m."""
    return (x + m - 1) // m * m


def pad_nvfp4_weight(
    weight: torch.Tensor,
    n_alignment: int = FP4_GEMM_ALIGNMENT,
    k_alignment: int = FP4_GEMM_ALIGNMENT,
) -> tuple[torch.Tensor, int]:
    """
    Pad packed NVFP4 weights to satisfy alignment constraints for FP4 GEMM kernels.

    Different backends have different alignment requirements:
    - CUTLASS/cuDNN: N % 32 == 0, K % 32 == 0
    - TRTLLM: N % 128 == 0 (for shuffle_matrix_sf_a), K padding handled separately

    Args:
        weight: Packed FP4 weight tensor of shape [N, K//2] (2 FP4 values per byte)
        n_alignment: Required alignment for N dimension (default 32, use 128 for TRTLLM)
        k_alignment: Required alignment for K dimension (default 32, use 0 to skip)

    Returns:
        Tuple of (padded_weight, weights_padding_cols) where weights_padding_cols
        is the number of columns added for K-dimension padding (in bytes).
    """
    weight_current_rows = weight.shape[0]  # N dimension
    weight_current_col_bytes = weight.shape[1]  # K//2 (packed)

    # Calculate padding for N dimension (rows)
    pad_rows = 0
    if n_alignment > 0 and weight_current_rows % n_alignment != 0:
        total_rows = round_up_to_multiple(weight_current_rows, n_alignment)
        pad_rows = total_rows - weight_current_rows

    # Calculate padding for K dimension (columns)
    # 2 FP4 items are packed per byte in the input dimension
    weight_current_col_elements = weight_current_col_bytes * 2
    pad_cols_bytes = 0
    if k_alignment > 0 and weight_current_col_elements % k_alignment != 0:
        total_cols = round_up_to_multiple(weight_current_col_elements, k_alignment)
        pad_cols = total_cols - weight_current_col_elements
        # pad_cols is in elements, but padding is in bytes (2 elements per byte)
        pad_cols_bytes = pad_cols // 2

    # Apply padding in a single operation if needed
    # For 2D tensor, pad argument is (pad_left, pad_right, pad_top, pad_bottom)
    if pad_rows > 0 or pad_cols_bytes > 0:
        weight = torch.nn.functional.pad(
            weight, (0, pad_cols_bytes, 0, pad_rows)
        ).contiguous()

    return weight, pad_cols_bytes


def pad_nvfp4_activation_for_cutlass(
    x_fp4: torch.Tensor,
    weights_padding_cols: int,
) -> torch.Tensor:
    """
    Pad packed FP4 activations to match the K-dimension padding applied to weights.

    Args:
        x_fp4: Packed FP4 activation tensor
        weights_padding_cols: Number of padding columns (in bytes) from weight padding

    Returns:
        Padded activation tensor
    """
    if weights_padding_cols > 0:
        return torch.nn.functional.pad(x_fp4, (0, weights_padding_cols)).contiguous()
    return x_fp4


def slice_nvfp4_output(
    out: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """
    Slice the output tensor to remove padding in N dimension if weight was padded.

    Args:
        out: Output tensor from FP4 GEMM
        output_size: Original output size before padding

    Returns:
        Sliced output tensor with padding removed
    """
    if out.shape[-1] != output_size:
        return out[..., :output_size].contiguous()
    return out
