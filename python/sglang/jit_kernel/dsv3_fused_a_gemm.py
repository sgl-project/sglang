"""
JIT kernel for DeepSeek V3 fused QKV-A GEMM (min-latency).

Runtime-compiled CUDA C++ kernel for SM90+ (Hopper) GPUs.
Shapes: hd_in a multiple of 256, hd_out a multiple of 16, num_tokens 1-16, bfloat16.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.common import direct_register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_dsv3_fused_a_gemm_module(hd_in: int, hd_out: int, use_pdl: bool) -> Module:
    args = make_cpp_args(hd_in, hd_out, use_pdl)
    return load_jit(
        "dsv3_fused_a_gemm",
        *args,
        cuda_files=["gemm/dsv3_fused_a_gemm.cuh"],
        cuda_wrappers=[
            ("dsv3_fused_a_gemm", f"DSV3FusedAGemmKernel<{args}>::run"),
        ],
    )


def _dsv3_fused_a_gemm_run(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    assert mat_a.stride(1) == 1, "mat_a must be row-major [M, K]"
    output = torch.empty(
        (mat_a.shape[0], mat_b.shape[1]),
        device=mat_a.device,
        dtype=mat_a.dtype,
    )
    module = _jit_dsv3_fused_a_gemm_module(
        mat_a.shape[1], mat_b.shape[1], is_arch_support_pdl()
    )
    module.dsv3_fused_a_gemm(mat_a, mat_b, output)
    return output


def _dsv3_fused_a_gemm_fake(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    return mat_a.new_empty((mat_a.shape[0], mat_b.shape[1]), dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="jit_dsv3_fused_a_gemm",
    op_func=_dsv3_fused_a_gemm_run,
    mutates_args=[],
    fake_impl=_dsv3_fused_a_gemm_fake,
)


@debug_kernel_api
def dsv3_fused_a_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    DeepSeek V3 fused QKV-A GEMM kernel (JIT variant).

    Args:
        mat_a: Input tensor of shape [num_tokens, hd_in], bfloat16, row-major.
            hd_in must be a multiple of 256 and num_tokens in [1, 16].
        mat_b: Weight tensor of shape [hd_in, hd_out], bfloat16, column-major
            (i.e. ``weight.T`` of a row-major [hd_out, hd_in] weight).
            hd_out must be a multiple of 16.
        output: Optional pre-allocated output tensor of shape [num_tokens, hd_out].

    Returns:
        Output tensor of shape [num_tokens, hd_out].
    """
    result = torch.ops.sglang.jit_dsv3_fused_a_gemm(mat_a, mat_b)
    if output is not None:
        output.copy_(result)
        return output
    return result
