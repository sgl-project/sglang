"""
JIT kernel for DeepSeek V3 fused QKV-A GEMM (min-latency).

Runtime-compiled CUDA C++ kernel for SM90+ (Hopper) GPUs.
Shapes: hd_in a multiple of 256, hd_out a multiple of 16, num_tokens 1-16, bfloat16.

``dsv3_fused_a_gemm_fp8`` is the W8A16 variant for block-FP8 weights (128x128 scales); SM103 only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.common import direct_register_custom_op
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def dsv3_fused_a_gemm_module(hd_in: int, hd_out: int, use_pdl: bool) -> Module:
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
    module = dsv3_fused_a_gemm_module(
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


def _is_sm103() -> bool:
    if is_hip_runtime():
        return False
    arch = get_jit_cuda_arch()
    return (arch.major, arch.minor) == (10, 3)


@cache_once
def dsv3_fused_a_gemm_fp8_module(k: int, n: int, scale_dtype: torch.dtype) -> Module:
    if not _is_sm103():
        raise RuntimeError("dsv3_fused_a_gemm_fp8 is validated on SM103 (GB300) only")
    if k <= 0 or k % 128 or n <= 0 or n % 16:
        raise ValueError("K must be a positive multiple of 128; N a multiple of 16")
    if scale_dtype not in (torch.float32, torch.int32):
        raise ValueError("weight_scale must contain FP32 scales or packed UE8M0 int32")
    args = make_cpp_args(k, n, scale_dtype == torch.int32, is_arch_support_pdl())
    return load_jit(
        "dsv3_fused_a_gemm_fp8",
        *args,
        cuda_files=["gemm/dsv3_fused_a_gemm_fp8.cuh"],
        cuda_wrappers=[("run", f"DSV3FusedAGemmFP8Kernel<{args}>::run")],
    )


@register_custom_op(
    op_name="jit_dsv3_fused_a_gemm_fp8", mutates_args=["out"], eager=True
)
def _dsv3_fused_a_gemm_fp8_run(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    module = dsv3_fused_a_gemm_fp8_module(
        weight.shape[1], weight.shape[0], weight_scale.dtype
    )
    module.run(input, weight, weight_scale, out)


@debug_kernel_api
def dsv3_fused_a_gemm_fp8(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``input[M,K] @ dequant(weight[N,K]).T``; BF16 input/output.

    FP8 weights are widened exactly in registers; K=128 FP32 partial sums are scaled
    per block. On the FP16-operand shapes the BF16 activations are converted to FP16
    in registers: exact for 2^-17 <= |x| <= 65504, saturated above that.

    Requires contiguous CUDA tensors, M in [1,16], K divisible by 128, and N
    divisible by 16. Weight is E4M3FN. Scales are either FP32
    ``[ceil(N/128), K/128]`` or DeepGEMM's packed UE8M0 int32
    ``[N, ceil(K/512)]``, with positive strides. Optional output must be
    contiguous BF16 ``[M,N]`` and must not alias the inputs.
    """
    if output is None:
        output = input.new_empty((input.shape[0], weight.shape[0]))
    _dsv3_fused_a_gemm_fp8_run(input, weight, weight_scale, output)
    return output
