"""Quantization kernels (per-token / per-token-group FP8 & INT8)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CUDA_ONLY,
    CapabilityRequirement,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch


register_kernel(
    KernelSpec(
        op="quantization.sgl_per_token_quant_fp8",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.quantization.per_token_quant_fp8:per_token_quant_fp8",
        capabilities=CUDA_ONLY,
    )
)
# fp8 / int8 are legacy aliases of the same 8bit kernel in the wheel.
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
        )
    )
del _name

register_kernel(
    KernelSpec(
        op="quantization.per_token_group_quant",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.quantization.per_token_group_quant:per_token_group_quant",
        capabilities=CUDA_ONLY,
    )
)


def sgl_per_token_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    """Per-token FP8 quantization, writing into ``output_q`` / ``output_s``."""
    return get_kernel("quantization.sgl_per_token_quant_fp8", KernelBackend.JIT)(
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


__all__ = [
    "sgl_per_token_quant_fp8",
    "sgl_per_token_group_quant_8bit",
]


# Triton / CuTe DSL kernels migrated into this group from
# srt/layers/quantization (RFC #29630, Phase 2.5); registered for inventory.
# Import them from their modules.
_TRITON_KERNELS = [
    ("fp8_kernel", "sglang_per_token_group_quant_fp8"),
    ("fp8_kernel", "sglang_per_token_quant_fp8"),
    ("fp8_kernel", "static_quant_fp8"),
    ("fp8_kernel", "dequant_group_fp8_to_bf16"),
    ("fp8_kernel", "per_tensor_quant_mla_fp8"),
    ("fp8_kernel", "per_token_group_quant_mla_deep_gemm_masked_fp8"),
    ("int8_kernel", "per_token_quant_int8"),
    ("int8_kernel", "per_token_group_quant_int8"),
    ("awq_triton", "awq_dequantize_triton"),
    ("mxfp8_amd_gfx95", "mxfp8_e4m3_quantize"),
    ("mxfp8_swizzled_triton", "swiglu_oai_mxfp8"),
    ("mxfp8_swizzled_triton", "swiglu_oai_mxfp8_deepgemm"),
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
        capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(10, 0))}),
    )
)


# Public entry points inventoried by logical operator group (RFC #29630).
register_kernel(
    KernelSpec(
        op="quantization.cvt_fp8_e4m3",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.quantization.fp8_cvt:cvt_fp8_e4m3",
        capabilities=CUDA_ONLY,
    )
)
