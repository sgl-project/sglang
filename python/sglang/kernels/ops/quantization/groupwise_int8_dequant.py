from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _groupwise_int8_dequant_kernel(
    weight,
    scales,
    output,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if WIDTH % 128 == 0:
        scale_offsets = offsets // 128
    else:
        # A partial final group restarts at the next row.
        rows = offsets // WIDTH
        groups = (offsets % WIDTH) // 128
        scale_offsets = rows * tl.cdiv(WIDTH, 128) + groups
    values = tl.load(weight + offsets, offsets < N, other=0).to(tl.float32)
    scale = tl.load(scales + scale_offsets, offsets < N, other=0)
    result = (values * scale).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(output + offsets, result, offsets < N)


def _launch_dequant(weight, scales, output):
    count = weight.numel()
    if count:
        block = 1024
        _groupwise_int8_dequant_kernel[(triton.cdiv(count, block),)](
            weight, scales, output, count, weight.shape[1], block
        )


@register_custom_op(op_name="groupwise_int8_dequant_validated", mutates_args=["output"])
def _dequant_into_validated(
    weight: torch.Tensor, scales: torch.Tensor, output: torch.Tensor
) -> None:
    """Internal path: admitted weights, BF16 output, all on the current GPU.

    The caller must validate the fixed weight metadata after loading and keep
    it unchanged across forwards. Component device moves may change storage.
    """
    _launch_dequant(weight, scales, output)


@register_custom_op(op_name="groupwise_int8_dequant", mutates_args=["output"])
def _dequant_into(
    weight: torch.Tensor, scales: torch.Tensor, output: torch.Tensor
) -> None:
    if weight.ndim != 2 or scales.ndim != 2:
        raise ValueError("Weight and scales must be two-dimensional")
    rows, width = weight.shape
    if weight.dtype != torch.int8 or scales.dtype != torch.float32:
        raise ValueError("Weight must be INT8 and scales must be FP32")
    if scales.shape != (rows, triton.cdiv(width, 128)):
        raise ValueError("Scales must have shape [rows, ceil(width / 128)]")
    if output.shape != weight.shape or output.dtype != torch.bfloat16:
        raise ValueError("Output must match the weight shape and have BF16 dtype")
    if not weight.is_cuda or not (weight.device == scales.device == output.device):
        raise ValueError("Weight, scales and output must be on the same CUDA device")
    if not (
        weight.is_contiguous() and scales.is_contiguous() and output.is_contiguous()
    ):
        raise ValueError("Weight, scales and output must be contiguous")
    with torch.cuda.device(weight.device):
        _launch_dequant(weight, scales, output)


def groupwise_int8_dequant(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize contiguous CUDA INT8 weights with FP32 scales, group size 128.

    Scales have shape [rows, ceil(width / 128)]. Multiplication is performed
    in FP32 before rounding once to BF16, including a partial final group.
    """
    output = torch.empty_like(weight, dtype=torch.bfloat16)
    _dequant_into(weight, scales, output)
    return output
