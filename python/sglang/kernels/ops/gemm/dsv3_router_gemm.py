"""
Router GEMM with FlashInfer fixed-shape ops and an SGLang JIT fallback.

The JIT fallback supports num_experts in {256, 384}, hidden_dim a multiple of
1024, and num_tokens 1-16 on SM90+ GPUs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from flashinfer import gemm as flashinfer_gemm

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_FLASHINFER_ROUTER_GEMM_OPS = {
    (6144, 256, torch.float32): flashinfer_gemm.mm_M1_16_K6144_N256,
    (7168, 128, torch.bfloat16): flashinfer_gemm.mm_M1_16_K7168_N128,
    (7168, 256, torch.float32): flashinfer_gemm.mm_M1_16_K7168_N256,
    (7168, 256, torch.bfloat16): flashinfer_gemm.mm_M1_16_K7168_N256_bf16,
    (7168, 384, torch.float32): flashinfer_gemm.mm_M1_16_K7168_N384,
    (7168, 384, torch.bfloat16): flashinfer_gemm.mm_M1_16_K7168_N384_bf16,
    (7168, 896, torch.float32): flashinfer_gemm.mm_M1_16_K7168_N896,
    (7168, 896, torch.bfloat16): flashinfer_gemm.mm_M1_16_K7168_N896_bf16,
}
_FLASHINFER_ROUTER_GEMM_SMS = frozenset({90, 100, 103, 107})


@cache_once
def dsv3_router_gemm_module(
    num_experts: int,
    hidden_dim: int,
    use_pdl: bool,
    out_float: bool,
) -> Module:
    args = make_cpp_args(num_experts, hidden_dim, use_pdl, out_float)
    return load_jit(
        "dsv3_router_gemm",
        *args,
        cuda_files=["gemm/dsv3_router_gemm.cuh"],
        cuda_wrappers=[
            ("dsv3_router_gemm", f"DSV3RouterGemmKernel<{args}>::run"),
        ],
    )


@register_custom_op(
    op_name="dsv3_router_gemm",
    mutates_args=["output"],
)
def _dsv3_router_gemm_custom_op(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    num_experts = router_weights.shape[0]
    hidden_dim = hidden_states.shape[1]
    out_float = output.dtype == torch.float32
    module = dsv3_router_gemm_module(
        num_experts, hidden_dim, is_arch_support_pdl(), out_float
    )
    module.dsv3_router_gemm(hidden_states, router_weights, output)
    return None


@debug_kernel_api
def dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Router GEMM with FlashInfer fixed-shape ops and an SGLang JIT fallback.

    Args:
        hidden_states: Input tensor of shape [num_tokens, hidden_dim], bfloat16.
            hidden_dim must be a multiple of 1024 and num_tokens in [1, 16].
        router_weights: Weight tensor of shape [num_experts, hidden_dim], bfloat16.
        out_dtype: Output dtype, either torch.bfloat16 or torch.float32.
        output: Optional pre-allocated output tensor.

    Returns:
        Output tensor of shape [num_tokens, num_experts].
    """
    if output is None:
        output = torch.empty(
            hidden_states.shape[0],
            router_weights.shape[0],
            device=hidden_states.device,
            dtype=out_dtype,
        )

    flashinfer_op = _FLASHINFER_ROUTER_GEMM_OPS.get(
        (hidden_states.shape[1], router_weights.shape[0], output.dtype)
    )
    if flashinfer_op is not None:
        major, minor = torch.cuda.get_device_capability(hidden_states.device)
        if major * 10 + minor in _FLASHINFER_ROUTER_GEMM_SMS:
            flashinfer_op(
                hidden_states,
                router_weights.T,
                output,
                launch_with_pdl=is_arch_support_pdl(),
            )
            return output

    _dsv3_router_gemm_custom_op(hidden_states, router_weights, output)
    return output
