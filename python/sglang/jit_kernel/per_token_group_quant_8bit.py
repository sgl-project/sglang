from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_per_token_group_quant_8bit_module() -> Module:
    return load_jit(
        "per_tensor_quant_fp8",
        cuda_files=["gemm/per_token_group_quant_8bit.cuh"],
        cuda_wrappers=[("per_token_group_quant_8bit", f"per_token_group_quant_8bit")],
    )


def per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool = False,
) -> None:
    """
    Per-tensor quantization to FP8 format.

    Args:
        input: Input tensor to quantize (float, half, or bfloat16)
        output_q: Output quantized tensor (fp8_e4m3)
        output_s: Output scale tensor (float scalar or 1D tensor with 1 element)
    """
    module = _jit_per_token_group_quant_8bit_module()
    module.per_token_group_quant_8bit(
        input.view(-1),
        output_q.view(-1),
        output_s.view(-1),
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
    )
