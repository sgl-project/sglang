"""Quantization kernels (per-token / per-token-group FP8 & INT8)."""

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
        op="quantization.sgl_per_token_quant_fp8",
        backend=KernelBackend.CUDA_AOT,
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
            backend=KernelBackend.CUDA_AOT,
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
        backend=KernelBackend.CUDA_JIT,
        target="sglang.jit_kernel.per_token_group_quant_8bit:per_token_group_quant_8bit",
        capability=_CUDA,
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
    return get_kernel("quantization.sgl_per_token_quant_fp8", KernelBackend.CUDA_AOT)(
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
    return get_kernel(
        "quantization.sgl_per_token_group_quant_8bit", KernelBackend.CUDA_AOT
    )(
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
