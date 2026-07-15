"""Quantization kernels (per-token / per-token-group FP8 & INT8)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    DeviceType,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

_CUDA = (CapabilityRequirement(device=DeviceType.CUDA),)

register_kernel(
    KernelSpec(
        op="quantization.sgl_per_token_quant_fp8",
        backend=KernelBackend.AOT,
        target="sgl_kernel:sgl_per_token_quant_fp8",
        format_signature=FormatSignature(
            supported_dtypes=("float8_e4m3fn",),
            in_place=True,
            description="per-token FP8 quantization into output_q/output_s",
        ),
        description="Per-token FP8 quantization (sgl_kernel wheel).",
    )
)
# fp8 / int8 are legacy aliases of the same 8bit kernel in the wheel; register
# each public name so runtime imports resolve to a stable spec.
for _name in (
    "sgl_per_token_group_quant_8bit",
    "sgl_per_token_group_quant_fp8",
    "sgl_per_token_group_quant_int8",
):
    register_kernel(
        KernelSpec(
            op=f"quantization.{_name}",
            backend=KernelBackend.AOT,
            target=f"sgl_kernel:{_name}",
            format_signature=FormatSignature(
                in_place=True,
                description="per-token-group 8-bit quantization",
            ),
            description=f"{_name} (sgl_kernel wheel).",
        )
    )
del _name

register_kernel(
    KernelSpec(
        op="quantization.sgl_per_token_group_quant_8bit",
        backend=KernelBackend.JIT,
        target="sglang.jit_kernel.per_token_group_quant_8bit:per_token_group_quant_8bit",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            in_place=True,
            description="per-token-group 8-bit quantization (JIT variant)",
        ),
        description="Per-token-group 8-bit quantization (sglang.jit_kernel).",
    )
)


def sgl_per_token_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    """Per-token FP8 quantization, writing into ``output_q`` / ``output_s``."""
    return get_kernel("quantization.sgl_per_token_quant_fp8", KernelBackend.AOT)(
        input, output_q, output_s
    )


def sgl_per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
) -> None:
    """Per-token-group 8-bit quantization, writing into ``output_q`` / ``output_s``."""
    return get_kernel("quantization.sgl_per_token_group_quant_8bit", KernelBackend.AOT)(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        fuse_silu_and_mul,
        masked_m,
        enable_v2,
    )


# Legacy aliases kept for source compatibility with existing call sites.
sgl_per_token_group_quant_fp8 = sgl_per_token_group_quant_8bit
sgl_per_token_group_quant_int8 = sgl_per_token_group_quant_8bit


__all__ = [
    "sgl_per_token_quant_fp8",
    "sgl_per_token_group_quant_8bit",
    "sgl_per_token_group_quant_fp8",
    "sgl_per_token_group_quant_int8",
]


# Triton / CuTe DSL kernels migrated into this group from
# srt/layers/quantization (RFC #29630, Phase 2.5); registered for inventory.
# Import them from their modules.
_TRITON_KERNELS = [
    ("fp8_kernel", "per_token_group_quant_8bit"),
    ("fp8_kernel", "sglang_per_token_group_quant_fp8"),
    ("fp8_kernel", "sglang_per_token_group_quant_8bit"),
    ("fp8_kernel", "sglang_per_token_quant_fp8"),
    ("fp8_kernel", "static_quant_fp8"),
    ("fp8_kernel", "w8a8_block_fp8_matmul"),
    ("fp8_kernel", "mxfp8_block_scaled_matmul_triton"),
    ("fp8_kernel", "per_tensor_quant_mla_fp8"),
    ("fp8_kernel", "per_token_group_quant_mla_deep_gemm_masked_fp8"),
    ("fp8_kernel", "per_token_group_quant_fp8_hopper_moe_mn_major"),
    ("fp8_kernel", "per_group_transpose"),
    ("fp8_kernel", "triton_scaled_mm"),
    ("int8_kernel", "per_token_quant_int8"),
    ("int8_kernel", "per_token_group_quant_int8"),
    ("int8_kernel", "w8a8_block_int8_matmul"),
    ("awq_triton", "awq_dequantize_triton"),
    ("awq_triton", "awq_gemm_triton"),
    ("mxfp8_amd_gfx95", "mxfp8_e4m3_quantize"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"quantization.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.quantization.{_mod}:{_fn}",
        )
    )
del _mod, _fn

register_kernel(
    KernelSpec(
        op="quantization.nvfp4_gemm_swiglu_nvfp4_quant",
        backend=KernelBackend.CUTE_DSL,
        target=(
            "sglang.kernels.ops.quantization.nvfp4_gemm_swiglu_nvfp4_quant"
            ":nvfp4_gemm_swiglu_nvfp4_quant"
        ),
        capabilities=(
            CapabilityRequirement(device=DeviceType.CUDA, min_cuda_arch=(10, 0)),
        ),
        description="Fused NVFP4 GEMM + SwiGLU + NVFP4 quant (CuTe DSL, SM100).",
    )
)
