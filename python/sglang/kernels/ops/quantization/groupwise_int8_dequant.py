from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _jit_module():
    return load_jit(
        "groupwise_int8_dequant",
        cuda_files=["quantization/groupwise_int8_dequant.cuh"],
        cuda_wrappers=[("groupwise_int8_dequant", "groupwise_int8_dequant")],
        extra_cuda_cflags=["--ftz=false"],
    )


@register_custom_op(op_name="groupwise_int8_dequant", mutates_args=["output"])
def _dequant_into(
    weight: torch.Tensor, scales: torch.Tensor, output: torch.Tensor
) -> None:
    _jit_module().groupwise_int8_dequant(output, weight, scales)


def groupwise_int8_dequant(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize contiguous CUDA INT8 weights with FP32 scales, group size 128.

    Scales have shape [rows, ceil(width / 128)]. Multiplication is performed
    in FP32 before rounding once to BF16, including a partial final group.
    """
    output = torch.empty_like(weight, dtype=torch.bfloat16)
    _dequant_into(weight, scales, output)
    return output
