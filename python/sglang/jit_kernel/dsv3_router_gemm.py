"""
JIT kernel for DeepSeek V3 router GEMM.

Replaces the AOT sgl_kernel.dsv3_router_gemm for SM90+ (Hopper) GPUs.
Supports num_experts in {256, 384} and hidden_dim=7168, num_tokens 1-16.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HIDDEN_DIM = 7168
_SUPPORTED_NUM_EXPERTS = (256, 384)


@cache_once
def _jit_dsv3_router_gemm_module(
    num_experts: int,
    use_pdl: bool,
    out_float: bool,
) -> Module:
    args = make_cpp_args(num_experts, _HIDDEN_DIM, use_pdl, out_float)
    return load_jit(
        "dsv3_router_gemm",
        *args,
        cuda_files=["gemm/dsv3_router_gemm.cuh"],
        cuda_wrappers=[
            ("dsv3_router_gemm", f"DSV3RouterGemmKernel<{args}>::run"),
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_dsv3_router_gemm(num_experts: int, hidden_dim: int) -> bool:
    """Check whether the JIT dsv3_router_gemm kernel can be used.

    Requires SM90+ (Hopper) CUDA GPU, supported num_experts, and hidden_dim=7168.
    """
    if is_hip_runtime():
        return False
    if get_jit_cuda_arch().major < 9:
        return False
    if hidden_dim != _HIDDEN_DIM:
        return False
    if num_experts not in _SUPPORTED_NUM_EXPERTS:
        return False
    use_pdl = is_arch_support_pdl()
    try:
        # Pre-compile both output variants so the first real call is warm.
        _jit_dsv3_router_gemm_module(num_experts, use_pdl, out_float=False)
        _jit_dsv3_router_gemm_module(num_experts, use_pdl, out_float=True)
        return True
    except Exception:
        return False


@debug_kernel_api
def dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    DeepSeek V3 router GEMM kernel (JIT variant).

    Args:
        hidden_states: Input tensor of shape [num_tokens, hidden_dim], bfloat16.
        router_weights: Weight tensor of shape [num_experts, hidden_dim], bfloat16.
        out_dtype: Output dtype, either torch.bfloat16 or torch.float32.
        output: Optional pre-allocated output tensor.

    Returns:
        Output tensor of shape [num_tokens, num_experts].
    """
    num_experts = router_weights.shape[0]
    out_float = out_dtype == torch.float32
    if output is None:
        output = torch.empty(
            hidden_states.shape[0],
            num_experts,
            device=hidden_states.device,
            dtype=out_dtype,
        )
    module = _jit_dsv3_router_gemm_module(num_experts, is_arch_support_pdl(), out_float)
    module.dsv3_router_gemm(hidden_states, router_weights, output)
    return output
