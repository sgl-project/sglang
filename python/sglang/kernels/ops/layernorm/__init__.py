"""Layer-normalization kernels.

Public wrappers forward to the AOT ``sgl_kernel`` implementation (the stable
wheel boundary), which supports the broadest set of shapes. The JIT CUDA
backend is also registered for inventory/comparison; because its signature and
return convention differ (in-place, ``None`` return, ``(input, weight, out,
eps)`` order), select it explicitly via
``select_kernel("layernorm.rmsnorm", backend=KernelBackend.CUDA_JIT)`` rather
than through these wrappers.
"""

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

_NORM_DTYPES = ("float16", "bfloat16")
_CUDA = CapabilityRequirement(requires_cuda=True)

register_kernel(
    KernelSpec(
        op="layernorm.rmsnorm",
        backend=KernelBackend.CUDA_AOT,
        target="sgl_kernel:rmsnorm",
        format_signature=FormatSignature(
            supported_dtypes=_NORM_DTYPES,
            description="out = (x / RMS(x)) * weight; returns tensor",
        ),
        description="RMS normalization (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="layernorm.rmsnorm",
        backend=KernelBackend.CUDA_JIT,
        target="sglang.jit_kernel.norm:rmsnorm",
        capability=_CUDA,
        format_signature=FormatSignature(
            supported_dtypes=_NORM_DTYPES,
            in_place=True,
            description="in-place RMS norm; signature (input, weight, out, eps)",
        ),
        description="RMS normalization (sglang.jit_kernel).",
    )
)
register_kernel(
    KernelSpec(
        op="layernorm.fused_add_rmsnorm",
        backend=KernelBackend.CUDA_AOT,
        target="sgl_kernel:fused_add_rmsnorm",
        format_signature=FormatSignature(
            supported_dtypes=_NORM_DTYPES,
            in_place=True,
            description="residual += x; x = RMSNorm(residual) * weight",
        ),
        description="Fused residual-add + RMS normalization (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="layernorm.fused_add_rmsnorm",
        backend=KernelBackend.CUDA_JIT,
        target="sglang.jit_kernel.norm:fused_add_rmsnorm",
        capability=_CUDA,
        format_signature=FormatSignature(
            supported_dtypes=_NORM_DTYPES,
            in_place=True,
            description="in-place fused residual-add + RMS norm",
        ),
        description="Fused residual-add + RMS normalization (sglang.jit_kernel).",
    )
)
register_kernel(
    KernelSpec(
        op="layernorm.gemma_rmsnorm",
        backend=KernelBackend.CUDA_AOT,
        target="sgl_kernel:gemma_rmsnorm",
        format_signature=FormatSignature(
            supported_dtypes=_NORM_DTYPES,
            description="out = (x / RMS(x)) * (weight + 1); returns tensor",
        ),
        description="Gemma-style RMS normalization (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="layernorm.gemma_fused_add_rmsnorm",
        backend=KernelBackend.CUDA_AOT,
        target="sgl_kernel:gemma_fused_add_rmsnorm",
        format_signature=FormatSignature(
            supported_dtypes=_NORM_DTYPES,
            in_place=True,
            description="residual += x; x = GemmaRMSNorm(residual) * (weight + 1)",
        ),
        description="Gemma-style fused residual-add + RMS normalization.",
    )
)


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """RMS normalization: ``out = (input / RMS(input)) * weight``."""
    return get_kernel("layernorm.rmsnorm", KernelBackend.CUDA_AOT)(
        input, weight, eps, out, enable_pdl
    )


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    """In-place fused residual add + RMS normalization."""
    return get_kernel("layernorm.fused_add_rmsnorm", KernelBackend.CUDA_AOT)(
        input, residual, weight, eps, enable_pdl
    )


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Gemma-style RMS normalization: ``out = (input / RMS(input)) * (weight + 1)``."""
    return get_kernel("layernorm.gemma_rmsnorm", KernelBackend.CUDA_AOT)(
        input, weight, eps, out, enable_pdl
    )


def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    """In-place Gemma-style fused residual add + RMS normalization."""
    return get_kernel("layernorm.gemma_fused_add_rmsnorm", KernelBackend.CUDA_AOT)(
        input, residual, weight, eps, enable_pdl
    )


__all__ = [
    "rmsnorm",
    "fused_add_rmsnorm",
    "gemma_rmsnorm",
    "gemma_fused_add_rmsnorm",
]
