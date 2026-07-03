"""GEMM and fused-GEMM kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

_CUDA = CapabilityRequirement(requires_cuda=True)

register_kernel(
    KernelSpec(
        op="gemm.fp8_scaled_mm",
        backend=KernelBackend.CUDA_AOT,
        target="sgl_kernel:fp8_scaled_mm",
        priority=10,
        format_signature=FormatSignature(
            supported_dtypes=("float8_e4m3fn",),
            description="C = (A_fp8 @ B_fp8) * scales_a * scales_b (+ bias)",
        ),
        description="FP8 scaled matmul (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.dsv3_fused_a_gemm",
        backend=KernelBackend.CUDA_AOT,
        target="sgl_kernel:dsv3_fused_a_gemm",
        priority=10,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description="DeepSeek-V3 fused QKV-A GEMM",
        ),
        description="DeepSeek-V3 fused-A GEMM (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.dsv3_fused_a_gemm",
        backend=KernelBackend.CUDA_JIT,
        target="sglang.jit_kernel.dsv3_fused_a_gemm:dsv3_fused_a_gemm",
        priority=5,
        capability=_CUDA,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description="DeepSeek-V3 fused QKV-A GEMM (drop-in with AOT signature)",
        ),
        description="DeepSeek-V3 fused-A GEMM (sglang.jit_kernel).",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.dsv3_router_gemm",
        backend=KernelBackend.CUDA_JIT,
        target="sglang.jit_kernel.dsv3_router_gemm:dsv3_router_gemm",
        priority=10,
        capability=_CUDA,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description="DeepSeek-V3 router GEMM; num_tokens in [1, 16]",
        ),
        description="DeepSeek-V3 router GEMM (sglang.jit_kernel, JIT-only).",
    )
)


def fp8_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 scaled matmul: ``(mat_a @ mat_b) * scales_a * scales_b (+ bias)``."""
    return get_kernel("gemm.fp8_scaled_mm", KernelBackend.CUDA_AOT)(
        mat_a, mat_b, scales_a, scales_b, out_dtype, bias
    )


def dsv3_fused_a_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DeepSeek-V3 fused QKV-A GEMM."""
    return get_kernel("gemm.dsv3_fused_a_gemm", KernelBackend.CUDA_AOT)(
        mat_a, mat_b, output
    )


def dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DeepSeek-V3 router GEMM (JIT-backed). ``out_dtype`` defaults to bfloat16."""
    impl = get_kernel("gemm.dsv3_router_gemm", KernelBackend.CUDA_JIT)
    if out_dtype is None:
        return impl(hidden_states, router_weights, output=output)
    return impl(hidden_states, router_weights, out_dtype, output)


__all__ = ["fp8_scaled_mm", "dsv3_fused_a_gemm", "dsv3_router_gemm"]
