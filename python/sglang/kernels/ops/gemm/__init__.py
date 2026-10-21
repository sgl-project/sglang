"""GEMM and fused-GEMM kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import _platform, get_kernel
from sglang.kernels.spec import (
    CUDA_ONLY,
    CapabilityRequirement,
    KernelBackend,
    KernelSpec,
    capabilities_satisfied,
)

if TYPE_CHECKING:
    import torch

    from sglang.kernels.ops.gemm.cutedsl_dual_gemm import DualGemmQuantMode

_SM90 = frozenset({CapabilityRequirement.cuda(min_sm=(9, 0), max_sm=(9, 0))})
_SM120 = frozenset({CapabilityRequirement.cuda(min_sm=(12, 0), max_sm=(12, 0))})
_SM12X = frozenset({CapabilityRequirement.cuda(min_sm=(12, 0), max_sm=(12, 9))})
_SM10X = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 9))})
_KDA_PACKAGE = "sglang.kernels.kda_kernels"


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

    # SM90 benchmarks show repeatable NVJet wins only for large prefill M with
    # either a wide output or a broad down projection. Keep decode, narrow TP8
    # projections, and shapes outside that measured envelope on the AOT kernel.
    return m >= 8192 and ((k >= 4096 and n >= 6144) or (k >= 7168 and n >= 5376))


class Fp8ScaledMMOp(BaseFusedOp):
    """FP8 GEMM with independent per-row A and per-column B scales."""

    op = "gemm.fp8_scaled_mm"
    priority = (KernelBackend.AOT, KernelBackend.TORCH)
    capabilities = {KernelBackend.AOT: CUDA_ONLY}

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
        # The torch path is a measured SM90 win for some shapes, not a
        # capability limit; elsewhere the AOT kernel always runs.
        if (
            backend is KernelBackend.AOT
            and capabilities_satisfied(_SM90, _platform())
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
        op="gemm.dual_gemm_swiglu",
        backend=KernelBackend.CUTE_DSL,
        target="sglang.kernels.ops.gemm.cutedsl_dual_gemm:dual_gemm_swiglu",
        capabilities=_SM10X,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.dual_gemm_swiglu_fp8",
        backend=KernelBackend.CUTE_DSL,
        target="sglang.kernels.ops.gemm.cutedsl_dual_gemm:dual_gemm_swiglu_fp8",
        capabilities=_SM10X,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.nvfp4_gemm_swiglu_nvfp4_quant",
        backend=KernelBackend.CUTE_DSL,
        target=(
            "sglang.kernels.ops.gemm.nvfp4_gemm_swiglu_nvfp4_quant"
            ":nvfp4_gemm_swiglu_nvfp4_quant"
        ),
        capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(10, 0))}),
    )
)
register_kernel(
    KernelSpec(
        op="gemm.bmm_fp8",
        backend=KernelBackend.FLASHINFER,
        target="sglang.kernels.ops.gemm.flashinfer_bmm_fp8:bmm_fp8",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.dsv3_fused_a_gemm",
        backend=KernelBackend.AOT,
        target="sgl_kernel:dsv3_fused_a_gemm",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.dsv3_fused_a_gemm",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.dsv3_fused_a_gemm:dsv3_fused_a_gemm",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.tiny_gemm",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.tiny_gemm:tiny_gemm_bf16",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.n128k512",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.small_gemm_bf16:n128k512_gemm_bf16",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.n32k5120",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.small_gemm_bf16:n32k5120_gemm_bf16",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.qwen3x_nvfp4",
        backend=KernelBackend.KDA,
        target=f"{_KDA_PACKAGE}.qwen3x_nvfp4_gemm:try_qwen3x_nvfp4_gemm",
        capabilities=_SM120,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.sm120_fp8_linear",
        backend=KernelBackend.KDA,
        target=f"{_KDA_PACKAGE}.sm120_fp8:try_sm120_fp8_linear",
        capabilities=_SM12X,
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


def dual_gemm_swiglu_fp8(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    output_scale: Optional[torch.Tensor] = None,
    quant_mode: Optional["DualGemmQuantMode"] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse FP8 gate/up projections, SwiGLU, and activation quantization."""
    if quant_mode is None:
        from .cutedsl_dual_gemm import DualGemmQuantMode

        quant_mode = DualGemmQuantMode.DYNAMIC_PER_TOKEN
    return get_kernel("gemm.dual_gemm_swiglu_fp8", KernelBackend.CUTE_DSL)(
        x,
        gate_up_weight,
        x_scale,
        gate_up_weight_scale,
        output_scale,
        quant_mode,
    )


def dual_gemm_swiglu(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
) -> torch.Tensor:
    """Fuse BF16/FP16 gate/up projections followed by SwiGLU."""
    return get_kernel("gemm.dual_gemm_swiglu", KernelBackend.CUTE_DSL)(
        x, gate_up_weight
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


def try_qwen3x_nvfp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: Optional[torch.Tensor],
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor | None:
    """Run the validated SM120 fast path, or return ``None`` for fallback."""
    return get_kernel("gemm.qwen3x_nvfp4", KernelBackend.KDA)(
        input, weight, input_sf, weight_sf, alpha, out_dtype, out_features
    )


def try_sm120_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    output_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor | None:
    """Run the best validated SM12x FP8 small-batch path, or return ``None``."""
    return get_kernel("gemm.sm120_fp8_linear", KernelBackend.KDA)(
        input,
        weight,
        input_scale,
        output_scale,
        bias,
    )


def kimi_k3_tiny_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Shape-tuned tiny GEMM for Kimi-K3, with the existing torch fallback."""
    from .kimi_k3 import kimi_k3_tiny_gemm as impl

    return impl(x, w)


__all__ = [
    "Fp8ScaledMMOp",
    "bmm_fp8",
    "dual_gemm_swiglu",
    "dual_gemm_swiglu_fp8",
    "fp8_scaled_mm",
    "kimi_k3_tiny_gemm",
    "try_qwen3x_nvfp4_gemm",
    "try_sm120_fp8_linear",
]


# Public entry points inventoried by logical operator group (RFC #29630).
register_kernel(
    KernelSpec(
        op="gemm.linear_bf16_fp32",
        backend=KernelBackend.TORCH,
        target="sglang.kernels.ops.gemm.bf16_fp32:_linear_bf16_fp32_cublas",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.wo_a_bf16_gemv",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.gemm.dsv4_wo_a:wo_a_bf16_gemv",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.wo_a_bf16_small_batch",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.gemm.dsv4_wo_a:wo_a_bf16_small_batch",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.wo_a_bf16_small_batch_mxfp8",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.gemm.dsv4_wo_a:wo_a_bf16_small_batch_mxfp8",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.hopper_bf16_gemv",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.hopper_bf16_gemv:hopper_bf16_gemv",
        capabilities=_SM90 | _SM12X,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.fp8_blockwise_scaled_mm",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.fp8_blockwise_gemm:fp8_blockwise_scaled_mm",
        capabilities=frozenset(
            {CapabilityRequirement.cuda(min_sm=(12, 0), max_sm=(12, 9))}
        ),
    )
)
# Mirrors SUPPORTED_CAPABILITIES in ops.gemm.convrot_int8 (pinned by a test).
_CONVROT_INT8 = frozenset(
    CapabilityRequirement.cuda(min_sm=sm, max_sm=sm)
    for sm in ((9, 0), (10, 0), (12, 0), (12, 1))
)
register_kernel(
    KernelSpec(
        op="gemm.convrot_rotate_quantize_activation",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.convrot_int8:convrot_rotate_quantize_activation",
        capabilities=_CONVROT_INT8,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.convrot_int8_fused_linear",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.convrot_int8:convrot_int8_fused_linear",
        capabilities=_CONVROT_INT8,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.convrot_int8_linear_prequant",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.convrot_int8:convrot_int8_linear_prequant",
        capabilities=_CONVROT_INT8,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.sm120_fp8_gemv",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.sm120_fp8_gemv:sm120_fp8_gemv",
        capabilities=frozenset(
            {CapabilityRequirement.cuda(min_sm=(12, 0), max_sm=(12, 9))}
        ),
    )
)
register_kernel(
    KernelSpec(
        op="gemm.nvfp4_moe_mtp_mma",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.sm120_nvfp4_moe:nvfp4_moe_mtp_mma",
        capabilities=_SM120,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.sm120_nvfp4_moe",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.sm120_nvfp4_moe:try_nvfp4_moe",
        capabilities=_SM120,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.gptq_marlin_gemm",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.gptq_marlin:gptq_marlin_gemm",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.rel_proj_small_t",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.inkling_rel_proj:rel_proj_small_t",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.fused_hc_mix",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.gemm.hc_mix:fused_hc_mix",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="gemm.w8a8_block_fp8_matmul_triton",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.gemm.fp8_kernel:w8a8_block_fp8_matmul_triton",
        capabilities=frozenset({CapabilityRequirement.CUDA, CapabilityRequirement.HIP}),
    )
)
register_kernel(
    KernelSpec(
        op="gemm.w8a8_block_fp8_matmul_deepgemm",
        backend=KernelBackend.DEEPGEMM,
        target="sglang.kernels.ops.gemm.fp8_kernel:w8a8_block_fp8_matmul_deepgemm",
        capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(9, 0))}),
    )
)
register_kernel(
    KernelSpec(
        op="gemm.w8a8_mxfp8_matmul_deepgemm",
        backend=KernelBackend.DEEPGEMM,
        target="sglang.kernels.ops.gemm.fp8_kernel:w8a8_mxfp8_matmul_deepgemm",
        capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(10, 0))}),
    )
)
register_kernel(
    KernelSpec(
        op="gemm.triton_scaled_mm",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.gemm.fp8_kernel:triton_scaled_mm",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.w8a8_block_int8_matmul",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.gemm.int8_kernel:w8a8_block_int8_matmul",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.muse_spark_router_gemm",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.muse_spark_router_gemm:router_gemm",
        capabilities=CUDA_ONLY,
    )
)
