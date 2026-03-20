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


# --- ModelOpt W4A16 AWQ (reference dequant / linear, TRT-LLM layout) ---


def modelopt_w4a16_dequant_ref(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize in TRT-LLM layout: 0th dim of weight (in) is quantized group-wise.

    weight: (in, out_packed) uint8
    weight_scale: (num_groups, out) float32 with num_groups = in/group_size
    """
    in_features, out_packed = weight.shape
    out_features = out_packed * 2
    device = weight.device
    dtype_f = weight_scale.dtype
    low = (weight & 0xF).to(dtype_f)
    low_signed = torch.where(low >= 8, low - 16.0, low)
    high = (weight >> 4).to(dtype_f)
    high_signed = torch.where(high >= 8, high - 16.0, high)
    unpacked = torch.stack([low_signed, high_signed], dim=-1).reshape(
        in_features, out_features
    )
    scale_ref = weight_scale.to(device).repeat_interleave(group_size, dim=0)[
        :in_features, :
    ]
    w_fp = (unpacked * scale_ref).to(out_dtype)
    return w_fp


def modelopt_w4a16_linear_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    pre_quant_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Reference: pre_quant_scale on input, then x @ w_fp + bias (TRT-LLM layout)."""
    if pre_quant_scale is not None:
        x = x * pre_quant_scale.to(x.dtype)
    w_fp = modelopt_w4a16_dequant_ref(
        weight, weight_scale, group_size, out_dtype=out_dtype or x.dtype
    )
    return torch.nn.functional.linear(x, w_fp.t(), bias)


def pre_quant_scale_sharded_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None:
    """Shard dim 0 when param is a TP partition; else copy full tensor."""
    from sglang.srt.layers.dp_attention import get_attention_tp_rank

    if param.size() == loaded_weight.size():
        param.data.copy_(loaded_weight)
        return
    tp_rank = get_attention_tp_rank()
    shard_size = param.data.shape[0]
    start_idx = tp_rank * shard_size
    param.data.copy_(loaded_weight.narrow(0, start_idx, shard_size))
