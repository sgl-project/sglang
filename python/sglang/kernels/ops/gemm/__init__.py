"""GEMM and fused-GEMM kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
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

_CUDA = frozenset({CapabilityRequirement.CUDA})
_SM90 = frozenset({CapabilityRequirement.cuda(min_sm=(9, 0), max_sm=(9, 0))})


def _prefer_torch_rowwise_fp8(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor],
) -> bool:
    """Whether Torch's SM90 NVJet kernel wins for this row/column-scaled shape."""
    import torch

    if (
        mat_a.device.type != "cuda"
        or mat_b.device != mat_a.device
        or not hasattr(torch, "_scaled_mm")
        or out_dtype != torch.bfloat16
        or bias is not None
        or mat_a.dtype != torch.float8_e4m3fn
        or mat_b.dtype != torch.float8_e4m3fn
        or mat_a.ndim != 2
        or mat_b.ndim != 2
        or mat_a.stride(1) != 1
        or mat_b.stride(0) != 1
    ):
        return False

    m, k = mat_a.shape
    n = mat_b.shape[1]
    # This path is intentionally row/column scaled, never tensorwise: A has
    # one independent FP32 scale per input row and B one per output column.
    if (
        scales_a.dtype != torch.float32
        or scales_b.dtype != torch.float32
        or scales_a.device != mat_a.device
        or scales_b.device != mat_a.device
        or not scales_a.is_contiguous()
        or not scales_b.is_contiguous()
        or scales_a.numel() != m
        or scales_b.numel() != n
    ):
        return False

    # Tuned on H100 over MiniMax-H3's complete dense shape set: four
    # production sequence lengths and TP1/2/4/8 (64 shapes). This selector
    # chose the measured winner for every shape while retaining the AOT kernel
    # for the smaller-K projections where NVJet loses.
    return (k >= 5376 and n >= 3584) or (k >= 3584 and m >= 8192)


class Fp8ScaledMMOp(BaseFusedOp):
    """FP8 GEMM with independent per-row A and per-column B scales."""

    op = "gemm.fp8_scaled_mm"
    priority = (KernelBackend.AOT, KernelBackend.TORCH)
    capabilities = {
        KernelBackend.AOT: _CUDA,
        KernelBackend.TORCH: _SM90,
    }
    format_signature = FormatSignature(
        supported_dtypes=("float8_e4m3fn",),
        description=(
            "C[M,N] = (A_fp8[M,K] @ B_fp8[K,N]) * scale_a[M,1] * scale_b[1,N] (+ bias)"
        ),
    )
    descriptions = {
        KernelBackend.AOT: "Row/column-scaled FP8 matmul (sgl_kernel wheel).",
        KernelBackend.TORCH: "Row/column-scaled FP8 matmul (Torch NVJet on SM90).",
    }

    def backend_eligible(
        self,
        backend: KernelBackend,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scales_a: torch.Tensor,
        scales_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
    ) -> bool:
        if (
            backend is KernelBackend.AOT
            and super().backend_eligible(
                KernelBackend.TORCH,
                mat_a,
                mat_b,
                scales_a,
                scales_b,
                out_dtype,
                bias,
            )
            and _prefer_torch_rowwise_fp8(
                mat_a, mat_b, scales_a, scales_b, out_dtype, bias
            )
        ):
            return False
        return super().backend_eligible(
            backend, mat_a, mat_b, scales_a, scales_b, out_dtype, bias
        )

    def forward_native(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scales_a: torch.Tensor,
        scales_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        import torch

        m, n = mat_a.shape[0], mat_b.shape[1]
        scale_a = scales_a.reshape(m, 1) if scales_a.numel() == m else scales_a
        scale_b = scales_b.reshape(1, n) if scales_b.numel() == n else scales_b
        return torch._scaled_mm(
            mat_a,
            mat_b,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=out_dtype,
            bias=bias,
            use_fast_accum=True,
        )

    def forward_aot(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scales_a: torch.Tensor,
        scales_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        import sgl_kernel

        return sgl_kernel.fp8_scaled_mm(
            mat_a, mat_b, scales_a, scales_b, out_dtype, bias
        )


_FP8_SCALED_MM = register_fused_op(Fp8ScaledMMOp(), __name__, "_FP8_SCALED_MM")

register_kernel(
    KernelSpec(
        op="gemm.bmm_fp8",
        backend=KernelBackend.FLASHINFER,
        target="sglang.srt.layers.quantization.fp8_utils:bmm_fp8",
        capabilities=_CUDA,
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
        backend=KernelBackend.AOT,
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
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.dsv3_fused_a_gemm:dsv3_fused_a_gemm",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description="DeepSeek-V3 fused QKV-A GEMM (drop-in with AOT signature)",
        ),
        description="DeepSeek-V3 fused-A GEMM (sglang.kernels.jit).",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.dsv3_router_gemm",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.dsv3_router_gemm:dsv3_router_gemm",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description="DeepSeek-V3 router GEMM; num_tokens in [1, 16]",
        ),
        description="DeepSeek-V3 router GEMM (sglang.kernels.jit, JIT-only).",
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
    """FP8 matmul with per-row A and per-column B scales."""
    return _FP8_SCALED_MM(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)


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
    return get_kernel("gemm.dsv3_fused_a_gemm", KernelBackend.AOT)(mat_a, mat_b, output)


def dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DeepSeek-V3 router GEMM (JIT-backed). ``out_dtype`` defaults to bfloat16."""
    impl = get_kernel("gemm.dsv3_router_gemm", KernelBackend.JIT)
    if out_dtype is None:
        return impl(hidden_states, router_weights, output=output)
    return impl(hidden_states, router_weights, out_dtype, output)


__all__ = [
    "Fp8ScaledMMOp",
    "fp8_scaled_mm",
    "bmm_fp8",
    "dsv3_fused_a_gemm",
    "dsv3_router_gemm",
]


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
