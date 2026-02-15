from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_dsv3_fused_a_gemm_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "dsv3_fused_a_gemm",
        *args,
        cuda_files=["gemm/dsv3_fused_a_gemm.cuh"],
        cuda_wrappers=[("dsv3_fused_a_gemm", "dsv3_fused_a_gemm")],
    )


@register_custom_op(
    op_name="jit_dsv3_fused_a_gemm",
    mutates_args=["output"],
)
def jit_dsv3_fused_a_gemm(
    output: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
) -> None:
    """
    Fused A GEMM kernel for DeepSeek V3 low-latency inference.

    Args:
        output: Output tensor [num_tokens, hd_out] (bf16, row-major)
        mat_a: Input tensor [num_tokens, hd_in] (bf16, row-major)
        mat_b: Weight tensor [hd_in, hd_out] (bf16, column-major, stride(0)==1)
    """
    module = _jit_dsv3_fused_a_gemm_module()
    module.dsv3_fused_a_gemm(output, mat_a, mat_b)


def dsv3_fused_a_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    High-level wrapper matching the sgl_kernel.dsv3_fused_a_gemm interface.

    Args:
        mat_a: Input tensor [num_tokens, hd_in] (bf16, row-major)
        mat_b: Weight tensor [hd_in, hd_out] (bf16, column-major)
        output: Optional pre-allocated output tensor

    Returns:
        Output tensor [num_tokens, hd_out]
    """
    if output is None:
        output = torch.empty(
            (mat_a.shape[0], mat_b.shape[1]),
            device=mat_a.device,
            dtype=mat_a.dtype,
        )
    jit_dsv3_fused_a_gemm(output, mat_a, mat_b)
    return output