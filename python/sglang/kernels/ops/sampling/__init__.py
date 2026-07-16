"""Sampling kernels (top-k / top-p probability renormalization)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

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
        op="sampling.top_k_renorm_probs",
        backend=KernelBackend.CUDA_AOT,
        target="sgl_kernel.sampling:top_k_renorm_probs",
        format_signature=FormatSignature(
            description="renormalize probs by top-k thresholding; returns tensor"
        ),
        description="Top-k probability renormalization (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="sampling.top_p_renorm_probs",
        backend=KernelBackend.CUDA_AOT,
        target="sgl_kernel.sampling:top_p_renorm_probs",
        format_signature=FormatSignature(
            description="renormalize probs by top-p thresholding; returns tensor"
        ),
        description="Top-p probability renormalization (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="sampling.fused_topk_topp_renorm",
        backend=KernelBackend.CUDA_JIT,
        target="sglang.jit_kernel.fused_topk_topp:fused_topk_topp_renorm",
        capability=_CUDA,
        format_signature=FormatSignature(
            description="fused top-k then top-p renorm; matches sequential renorm"
        ),
        description="Fused top-k + top-p renorm (sglang.jit_kernel).",
    )
)


def top_k_renorm_probs(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    """Renormalize ``probs`` by top-k thresholding."""
    return get_kernel("sampling.top_k_renorm_probs", KernelBackend.CUDA_AOT)(
        probs, top_k
    )


def top_p_renorm_probs(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Renormalize ``probs`` by top-p thresholding."""
    return get_kernel("sampling.top_p_renorm_probs", KernelBackend.CUDA_AOT)(
        probs, top_p
    )


def fused_topk_topp_renorm(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    workspace: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused top-k + top-p probability renormalization (JIT CUDA)."""
    return get_kernel("sampling.fused_topk_topp_renorm", KernelBackend.CUDA_JIT)(
        probs, top_ks, top_ps, workspace=workspace, out=out
    )


def is_fused_topk_topp_available() -> bool:
    """True when the fused JIT kernel can be loaded."""
    try:
        from sglang.jit_kernel.fused_topk_topp import (
            is_fused_topk_topp_available as _available,
        )

        return _available()
    except Exception:
        return False


__all__ = [
    "top_k_renorm_probs",
    "top_p_renorm_probs",
    "fused_topk_topp_renorm",
    "is_fused_topk_topp_available",
]


# Migrated from srt/layers/utils/hash.py (RFC #29630, Phase 2.5).
register_kernel(
    KernelSpec(
        op="sampling.murmur_hash32",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.sampling.murmur_hash:murmur_hash32",
    )
)
