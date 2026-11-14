"""Ascend arch35 G-B FP8 GEMMs with checkpoint-format FP32 block scales.

Weights remain in [N, K] (or [E, N, K]) checkpoint layout. Only transpose
views are passed to CANN; neither weights nor scales are requantized to MX.
The TP deployment path requires nonempty activation batches; individual expert
groups may be empty.
"""

from typing import Optional

import torch


def _quantize_input(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.npu.npu_dynamic_block_quant(
        x, dst_type=torch.float8_e4m3fn, row_block_size=1, col_block_size=128
    )


def npu_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BF16/FP16 input, 128x128 FP8 weights, and FP32 block scales."""
    output_dtype = input.dtype
    output_shape = (*input.shape[:-1], weight.shape[0])
    x = input.reshape(-1, input.shape[-1]).contiguous()
    qx, x_scale = _quantize_input(x)
    output = torch.ops.npu.npu_quant_matmul(
        qx,
        weight.t(),
        scale=weight_scale.t(),
        pertoken_scale=x_scale,
        output_dtype=output_dtype,
        group_sizes=[1, 128, 128],
    )
    # CANN's FP8 G-B mode does not accept bias. Add it after the GEMM.
    if bias is not None:
        output = output + bias.to(output_dtype)
    return output.reshape(output_shape)


def npu_w8a8_block_fp8_grouped_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """BF16/FP16 grouped input, [E, N, K] FP8 weights, and FP32 scales."""
    qx, x_scale = _quantize_input(input.contiguous())
    # G-B mode is inferred from the FP32 [E, K/128, N/128] weight scale
    # and [M, K/128] input scale; GMM has no group_sizes parameter.
    return torch.ops.npu.npu_grouped_matmul(
        x=[qx],
        weight=[weight.transpose(-1, -2)],
        scale=[weight_scale.transpose(-1, -2)],
        per_token_scale=[x_scale],
        group_list=group_list,
        group_list_type=group_list_type,
        group_type=0,
        split_item=2,
        output_dtype=output_dtype,
    )[0]
