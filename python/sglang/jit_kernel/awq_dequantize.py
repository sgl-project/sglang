from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_awq_dequantize_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "awq_dequantize",
        *args,
        cuda_files=["gemm/awq_dequantize.cuh"],
        cuda_wrappers=[("awq_dequantize", f"awq_dequantize<{args}>")],
    )


def awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
) -> torch.Tensor:
    qweight_rows = qweight.shape[0]
    qweight_cols = qweight.shape[1]
    output = torch.empty(
        (qweight_rows, qweight_cols * 8),
        dtype=scales.dtype,
        device=scales.device,
    )
    module = _jit_awq_dequantize_module(scales.dtype)
    module.awq_dequantize(output, qweight, scales, qzeros)
    return output
