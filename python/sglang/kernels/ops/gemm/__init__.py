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
        format_signature=FormatSignature(
            supported_dtypes=("float8_e4m3fn",),
            description="C = (A_fp8 @ B_fp8) * scales_a * scales_b (+ bias)",
        ),
        description="FP8 scaled matmul (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.bmm_fp8",
        backend=KernelBackend.FLASHINFER,
        target="sglang.srt.layers.quantization.fp8_utils:bmm_fp8",
        capability=_CUDA,
        format_signature=FormatSignature(
            supported_dtypes=("float8_e4m3fn", "float8_e5m2"),
            description="batched (3D) per-tensor-scale FP8 matmul: D = A_fp8 @ B_fp8 * A_scale * B_scale",
        ),
        description="Batched FP8 matmul (flashinfer cuBLAS backend, torch.compile-safe wrapper).",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.dsv3_fused_a_gemm",
        backend=KernelBackend.CUDA_AOT,
        target="sgl_kernel:dsv3_fused_a_gemm",
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


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched (3D) per-tensor-scale FP8 matmul, via flashinfer's cuBLAS backend."""
    return get_kernel("gemm.bmm_fp8", KernelBackend.FLASHINFER)(
        A, B, A_scale, B_scale, dtype, out
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


__all__ = ["fp8_scaled_mm", "bmm_fp8", "dsv3_fused_a_gemm", "dsv3_router_gemm"]


# LoRA SGMV Triton kernels migrated into this group (from lora/triton_ops);
# registered for inventory. Import them from their modules.
_TRITON_KERNELS = [
    ("chunked_embedding_lora_a", "chunked_embedding_lora_a_forward"),
    ("chunked_sgmv_expand", "chunked_sgmv_lora_expand_forward"),
    ("chunked_sgmv_shrink", "chunked_sgmv_lora_shrink_forward"),
    ("embedding_lora_a", "embedding_lora_a_fwd"),
    ("gate_up_lora_b", "gate_up_lora_b_fwd"),
    ("qkv_lora_b", "qkv_lora_b_fwd"),
    ("sgemm_lora_a", "sgemm_lora_a_fwd"),
    ("sgemm_lora_b", "sgemm_lora_b_fwd"),
    ("kv_b_lora_absorbed", "step_a_q_fwd"),
    ("kv_b_lora_absorbed", "step_b_q_fwd"),
    ("kv_b_lora_absorbed", "step_a_v_fwd"),
    ("kv_b_lora_absorbed", "step_b_v_fwd"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"gemm.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.gemm.{_mod}:{_fn}",
        )
    )
del _mod, _fn
