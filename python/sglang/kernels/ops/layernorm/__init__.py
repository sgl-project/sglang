"""Layer-normalization kernels.

Each operator is a :class:`~sglang.kernels.fused_op.BaseFusedOp` with a
pure-``torch`` reference (``forward_native``) plus optimized CUDA backends,
all behind one signature. The public module-level functions are thin wrappers
over module-level instances; auto-selection prefers the AOT ``sgl_kernel``
implementation on CUDA and falls back to the native reference elsewhere.
Pick a specific backend with e.g.
``_RMSNORM.forward(x, w, backend=KernelBackend.CUDA_JIT)`` or globally via
``SGLANG_FORCE_FUSED_OP_BACKEND``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
)

if TYPE_CHECKING:
    import torch

_NORM_DTYPES = ("float16", "bfloat16")
_CUDA = CapabilityRequirement(requires_cuda=True)
_NORM_PRIORITY = (
    KernelBackend.CUDA_AOT,
    KernelBackend.CUDA_JIT,
    KernelBackend.TORCH,
)


class RMSNormOp(BaseFusedOp):
    """``out = (input / RMS(input)) * weight``; returns a tensor.

    ``enable_pdl`` is honored by the AOT backend only.
    """

    op = "layernorm.rmsnorm"
    priority = _NORM_PRIORITY
    capabilities = {
        KernelBackend.CUDA_AOT: _CUDA,
        KernelBackend.CUDA_JIT: _CUDA,
    }
    format_signature = FormatSignature(
        supported_dtypes=_NORM_DTYPES,
        description="out = (x / RMS(x)) * weight; returns tensor",
    )
    descriptions = {
        KernelBackend.CUDA_AOT: "RMS normalization (sgl_kernel wheel).",
        KernelBackend.CUDA_JIT: "RMS normalization (sglang.jit_kernel).",
        KernelBackend.TORCH: "RMS normalization (pure-torch reference).",
    }

    def forward_native(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import torch

        x = input.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        result = (x * weight).to(input.dtype)
        if out is None:
            return result
        out.copy_(result)
        return out

    def forward_cuda_aot(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import sgl_kernel

        return sgl_kernel.rmsnorm(input, weight, eps, out, enable_pdl)

    def forward_cuda_jit(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import torch

        from sglang.jit_kernel.norm import rmsnorm as jit_rmsnorm

        if out is None:
            out = torch.empty_like(input)
        jit_rmsnorm(input, weight, out, eps)
        return out


class FusedAddRMSNormOp(BaseFusedOp):
    """In-place ``residual += input; input = RMSNorm(residual) * weight``.

    Writes the sum into ``residual`` and the normalized value into ``input``;
    returns ``None``. ``enable_pdl`` is honored by the AOT backend only.
    """

    op = "layernorm.fused_add_rmsnorm"
    priority = _NORM_PRIORITY
    capabilities = {
        KernelBackend.CUDA_AOT: _CUDA,
        KernelBackend.CUDA_JIT: _CUDA,
    }
    format_signature = FormatSignature(
        supported_dtypes=_NORM_DTYPES,
        in_place=True,
        description="residual += x; x = RMSNorm(residual) * weight",
    )
    descriptions = {
        KernelBackend.CUDA_AOT: (
            "Fused residual-add + RMS normalization (sgl_kernel wheel)."
        ),
        KernelBackend.CUDA_JIT: (
            "Fused residual-add + RMS normalization (sglang.jit_kernel)."
        ),
        KernelBackend.TORCH: (
            "Fused residual-add + RMS normalization (pure-torch reference)."
        ),
    }

    def forward_native(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import torch

        acc = input.to(torch.float32) + residual.to(torch.float32)
        residual.copy_(acc.to(residual.dtype))
        variance = acc.pow(2).mean(dim=-1, keepdim=True)
        normed = acc * torch.rsqrt(variance + eps)
        input.copy_((normed * weight).to(input.dtype))

    def forward_cuda_aot(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import sgl_kernel

        return sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)

    def forward_cuda_jit(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        from sglang.jit_kernel.norm import fused_add_rmsnorm as jit_fused_add_rmsnorm

        return jit_fused_add_rmsnorm(input, residual, weight, eps)


class GemmaRMSNormOp(BaseFusedOp):
    """``out = (input / RMS(input)) * (weight + 1)``; returns a tensor."""

    op = "layernorm.gemma_rmsnorm"
    priority = _NORM_PRIORITY
    capabilities = {KernelBackend.CUDA_AOT: _CUDA}
    format_signature = FormatSignature(
        supported_dtypes=_NORM_DTYPES,
        description="out = (x / RMS(x)) * (weight + 1); returns tensor",
    )
    descriptions = {
        KernelBackend.CUDA_AOT: "Gemma-style RMS normalization (sgl_kernel wheel).",
        KernelBackend.TORCH: "Gemma-style RMS normalization (pure-torch reference).",
    }

    def forward_native(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import torch

        x = input.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        result = (x * (1.0 + weight.to(torch.float32))).to(input.dtype)
        if out is None:
            return result
        out.copy_(result)
        return out

    def forward_cuda_aot(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import sgl_kernel

        return sgl_kernel.gemma_rmsnorm(input, weight, eps, out, enable_pdl)


class GemmaFusedAddRMSNormOp(BaseFusedOp):
    """In-place ``residual += input; input = GemmaRMSNorm(residual) * (weight + 1)``."""

    op = "layernorm.gemma_fused_add_rmsnorm"
    priority = _NORM_PRIORITY
    capabilities = {KernelBackend.CUDA_AOT: _CUDA}
    format_signature = FormatSignature(
        supported_dtypes=_NORM_DTYPES,
        in_place=True,
        description="residual += x; x = GemmaRMSNorm(residual) * (weight + 1)",
    )
    descriptions = {
        KernelBackend.CUDA_AOT: ("Gemma-style fused residual-add + RMS normalization."),
        KernelBackend.TORCH: (
            "Gemma-style fused residual-add + RMS normalization "
            "(pure-torch reference)."
        ),
    }

    def forward_native(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import torch

        acc = input.to(torch.float32) + residual.to(torch.float32)
        residual.copy_(acc.to(residual.dtype))
        variance = acc.pow(2).mean(dim=-1, keepdim=True)
        normed = acc * torch.rsqrt(variance + eps)
        input.copy_((normed * (1.0 + weight.to(torch.float32))).to(input.dtype))

    def forward_cuda_aot(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import sgl_kernel

        return sgl_kernel.gemma_fused_add_rmsnorm(
            input, residual, weight, eps, enable_pdl
        )


_RMSNORM = register_fused_op(RMSNormOp(), __name__, "_RMSNORM")
_FUSED_ADD_RMSNORM = register_fused_op(
    FusedAddRMSNormOp(), __name__, "_FUSED_ADD_RMSNORM"
)
_GEMMA_RMSNORM = register_fused_op(GemmaRMSNormOp(), __name__, "_GEMMA_RMSNORM")
_GEMMA_FUSED_ADD_RMSNORM = register_fused_op(
    GemmaFusedAddRMSNormOp(), __name__, "_GEMMA_FUSED_ADD_RMSNORM"
)


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """RMS normalization: ``out = (input / RMS(input)) * weight``."""
    return _RMSNORM(input, weight, eps, out, enable_pdl)


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    """In-place fused residual add + RMS normalization."""
    return _FUSED_ADD_RMSNORM(input, residual, weight, eps, enable_pdl)


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Gemma-style RMS normalization: ``out = (input / RMS(input)) * (weight + 1)``."""
    return _GEMMA_RMSNORM(input, weight, eps, out, enable_pdl)


def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    """In-place Gemma-style fused residual add + RMS normalization."""
    return _GEMMA_FUSED_ADD_RMSNORM(input, residual, weight, eps, enable_pdl)


__all__ = [
    "RMSNormOp",
    "FusedAddRMSNormOp",
    "GemmaRMSNormOp",
    "GemmaFusedAddRMSNormOp",
    "rmsnorm",
    "fused_add_rmsnorm",
    "gemma_rmsnorm",
    "gemma_fused_add_rmsnorm",
]


from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelSpec

# Triton / TileLang kernels migrated from srt/layers top-level strays
# (RFC #29630, Phase 2.5); registered for inventory.
_PHASE25_KERNELS = [
    ("elementwise", "fused_dual_residual_rmsnorm", "triton"),
    ("elementwise", "fused_rmsnorm", "triton"),
    ("gemma4_fused_ops", "gemma4_fused_routing", "triton"),
    ("gemma4_fused_ops", "gemma_qkv_rmsnorm", "triton"),
    ("mhc_head", "fused_hc_head", "triton"),
]
for _mod, _fn, _bk in _PHASE25_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"layernorm.{_fn}",
            backend=KernelBackend(_bk),
            target=f"sglang.kernels.ops.layernorm.{_mod}:{_fn}",
        )
    )
del _mod, _fn, _bk
