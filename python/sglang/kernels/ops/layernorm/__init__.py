"""Layer-normalization kernels.

Each operator is a :class:`~sglang.kernels.fused_op.BaseFusedOp` with a
pure-``torch`` reference (``forward_native``) plus optimized per-device backends,
all behind one signature. The public module-level functions are thin wrappers
over module-level instances; auto-selection follows the production default for
the live device: AOT ``sgl_kernel`` on CUDA, ``aiter`` (or rocm-triton for
gemma) on ROCm, ``torch_npu`` on Ascend, native reference otherwise.
Pick a specific backend with e.g.
``_RMSNORM.forward(x, w, backend=KernelBackend.JIT)`` or globally via
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
_CUDA = frozenset({CapabilityRequirement.CUDA})
_HIP = frozenset({CapabilityRequirement.HIP})
_NPU = frozenset({CapabilityRequirement.NPU})
# Unlike the gated-activation ops, sgl_kernel does *not* build the rmsnorm ops
# for ROCm (production: ``if _is_cuda or _is_xpu or _is_musa: from sgl_kernel
# import rmsnorm`` — HIP is absent), so AOT here is CUDA-only. ROCm instead has
# an ``aiter`` path, and Ascend a ``torch_npu`` path — a clean illustration that
# the same ``AOT`` provenance covers different devices per op.
# Priority (best -> fallback) is device-agnostic; per-op CapabilityRequirement
# decides eligibility, so on CUDA this resolves to AOT, on HIP to AITER, on NPU
# to TORCH_NPU, each matching the production default for that device.
_NORM_PRIORITY = (
    KernelBackend.AOT,
    KernelBackend.JIT,
    KernelBackend.AITER,
    KernelBackend.TORCH_NPU,
    KernelBackend.TORCH,
)


class RMSNormOp(BaseFusedOp):
    """``out = (input / RMS(input)) * weight``; returns a tensor.

    ``enable_pdl`` is honored by the AOT backend only.
    """

    op = "layernorm.rmsnorm"
    priority = _NORM_PRIORITY
    capabilities = {
        KernelBackend.AOT: _CUDA,
        KernelBackend.JIT: _CUDA,
        KernelBackend.AITER: _HIP,
        KernelBackend.TORCH_NPU: _NPU,
    }
    format_signature = FormatSignature(
        supported_dtypes=_NORM_DTYPES,
        description="out = (x / RMS(x)) * weight; returns tensor",
    )
    descriptions = {
        KernelBackend.AOT: "RMS normalization (sgl_kernel wheel).",
        KernelBackend.JIT: "RMS normalization (sglang.jit_kernel).",
        KernelBackend.AITER: "RMS normalization (aiter rmsnorm2d_fwd, ROCm).",
        KernelBackend.TORCH_NPU: "RMS normalization (torch_npu, Ascend).",
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

    def forward_aot(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import sgl_kernel

        return sgl_kernel.rmsnorm(input, weight, eps, out, enable_pdl)

    def forward_jit(
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

    def forward_aiter(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import torch
        from aiter import rmsnorm2d_fwd

        # Mirrors production srt/layers/layernorm.py: rmsnorm2d_fwd(out, x, w, eps)
        # writes the normalized result in-place into ``out`` (ROCm path).
        if out is None:
            out = torch.empty_like(input)
        rmsnorm2d_fwd(out, input, weight, eps)
        return out

    def forward_npu(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import torch_npu

        result = torch_npu.npu_rms_norm(input, weight, eps)[0]
        if out is None:
            return result
        out.copy_(result)
        return out


class FusedAddRMSNormOp(BaseFusedOp):
    """In-place ``residual += input; input = RMSNorm(residual) * weight``.

    Writes the sum into ``residual`` and the normalized value into ``input``;
    returns ``None``. ``enable_pdl`` is honored by the AOT backend only.
    """

    op = "layernorm.fused_add_rmsnorm"
    priority = _NORM_PRIORITY
    capabilities = {
        KernelBackend.AOT: _CUDA,
        KernelBackend.JIT: _CUDA,
        KernelBackend.AITER: _HIP,
        KernelBackend.TORCH_NPU: _NPU,
    }
    format_signature = FormatSignature(
        supported_dtypes=_NORM_DTYPES,
        in_place=True,
        description="residual += x; x = RMSNorm(residual) * weight",
    )
    descriptions = {
        KernelBackend.AOT: (
            "Fused residual-add + RMS normalization (sgl_kernel wheel)."
        ),
        KernelBackend.JIT: (
            "Fused residual-add + RMS normalization (sglang.jit_kernel)."
        ),
        KernelBackend.AITER: ("Fused residual-add + RMS normalization (aiter, ROCm)."),
        KernelBackend.TORCH_NPU: (
            "Fused residual-add + RMS normalization (torch_npu, Ascend)."
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

    def forward_aot(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import sgl_kernel

        return sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)

    def forward_jit(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        from sglang.jit_kernel.norm import fused_add_rmsnorm as jit_fused_add_rmsnorm

        return jit_fused_add_rmsnorm(input, residual, weight, eps)

    def forward_aiter(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import torch
        from aiter import rmsnorm2d_fwd_with_add

        # aiter writes the normalized value and the new residual into separate
        # out buffers (production call order: out, x, residual_out, residual, w,
        # eps); copy them back to honor this op's in-place contract.
        out = torch.empty_like(input)
        residual_out = torch.empty_like(residual)
        rmsnorm2d_fwd_with_add(out, input, residual_out, residual, weight, eps)
        input.copy_(out)
        residual.copy_(residual_out)

    def forward_npu(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import torch_npu

        # torch_npu.npu_add_rms_norm(residual, x, w, eps) -> (normed, _, new_sum)
        out, _, residual_out = torch_npu.npu_add_rms_norm(residual, input, weight, eps)
        input.copy_(out)
        residual.copy_(residual_out)


class GemmaRMSNormOp(BaseFusedOp):
    """``out = (input / RMS(input)) * (weight + 1)``; returns a tensor."""

    op = "layernorm.gemma_rmsnorm"
    priority = _NORM_PRIORITY
    # AOT (sgl_kernel) on CUDA; JIT is the ROCm rocm-triton path
    # (sglang.jit_kernel.minimax_m3) — a JIT provenance pinned to HIP, distinct
    # from the CUDA-only JIT on the plain rmsnorm ops; torch_npu on Ascend.
    capabilities = {
        KernelBackend.AOT: _CUDA,
        KernelBackend.JIT: _HIP,
        KernelBackend.TORCH_NPU: _NPU,
    }
    format_signature = FormatSignature(
        supported_dtypes=_NORM_DTYPES,
        description="out = (x / RMS(x)) * (weight + 1); returns tensor",
    )
    descriptions = {
        KernelBackend.AOT: "Gemma-style RMS normalization (sgl_kernel wheel).",
        KernelBackend.JIT: (
            "Gemma-style RMS normalization (rocm-triton, sglang.jit_kernel)."
        ),
        KernelBackend.TORCH_NPU: ("Gemma-style RMS normalization (torch_npu, Ascend)."),
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

    def forward_aot(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import sgl_kernel

        return sgl_kernel.gemma_rmsnorm(input, weight, eps, out, enable_pdl)

    def forward_jit(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        from sglang.jit_kernel.minimax_m3.rmsnorm import (
            gemma_rmsnorm as rocm_triton_gemma_rmsnorm,
        )

        result = rocm_triton_gemma_rmsnorm(input, weight, eps)
        if out is None:
            return result
        out.copy_(result)
        return out

    def forward_npu(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import torch_npu

        result = torch_npu.npu_gemma_rms_norm(input, weight, eps)[0]
        if out is None:
            return result
        out.copy_(result)
        return out


class GemmaFusedAddRMSNormOp(BaseFusedOp):
    """In-place ``residual += input; input = GemmaRMSNorm(residual) * (weight + 1)``."""

    op = "layernorm.gemma_fused_add_rmsnorm"
    priority = _NORM_PRIORITY
    # AOT (sgl_kernel) on CUDA; JIT is the ROCm rocm-triton path on HIP.
    # NPU here would use ``sgl_kernel_npu.add_gemma_rms_norm`` (a distinct AOT-npu
    # wheel provenance, not torch_npu) — deferred until that provenance lands.
    capabilities = {
        KernelBackend.AOT: _CUDA,
        KernelBackend.JIT: _HIP,
    }
    format_signature = FormatSignature(
        supported_dtypes=_NORM_DTYPES,
        in_place=True,
        description="residual += x; x = GemmaRMSNorm(residual) * (weight + 1)",
    )
    descriptions = {
        KernelBackend.AOT: ("Gemma-style fused residual-add + RMS normalization."),
        KernelBackend.JIT: (
            "Gemma-style fused residual-add + RMS normalization "
            "(rocm-triton, sglang.jit_kernel)."
        ),
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

    def forward_aot(
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

    def forward_jit(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        from sglang.jit_kernel.minimax_m3.rmsnorm import (
            gemma_fused_add_rmsnorm as rocm_triton_gemma_fused_add_rmsnorm,
        )

        # rocm-triton returns (normed, new_residual); honor the in-place contract.
        norm_out, residual_out = rocm_triton_gemma_fused_add_rmsnorm(
            input, residual, weight, eps
        )
        input.copy_(norm_out)
        residual.copy_(residual_out)


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
