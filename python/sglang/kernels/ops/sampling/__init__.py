"""Sampling kernels (top-k / top-p probability renormalization)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

_CUDA = frozenset({CapabilityRequirement.CUDA})
_HIP = frozenset({CapabilityRequirement.HIP})

if TYPE_CHECKING:
    import torch

register_kernel(
    KernelSpec(
        op="sampling.top_k_renorm_probs",
        backend=KernelBackend.AOT,
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
        backend=KernelBackend.AOT,
        target="sgl_kernel.sampling:top_p_renorm_probs",
        format_signature=FormatSignature(
            description="renormalize probs by top-p thresholding; returns tensor"
        ),
        description="Top-p probability renormalization (sgl_kernel wheel).",
    )
)


def top_k_renorm_probs(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    """Renormalize ``probs`` by top-k thresholding."""
    return get_kernel("sampling.top_k_renorm_probs", KernelBackend.AOT)(probs, top_k)


def top_p_renorm_probs(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Renormalize ``probs`` by top-p thresholding."""
    return get_kernel("sampling.top_p_renorm_probs", KernelBackend.AOT)(probs, top_p)


__all__ = ["top_k_renorm_probs", "top_p_renorm_probs"]


# Migrated from srt/layers/utils/hash.py (RFC #29630, Phase 2.5).
# One backend per device: JIT on CUDA, Triton on ROCm. The ``capabilities``
# make exactly one spec eligible per platform (so ``backend=`` only selects
# among the specs eligible on the current device). The ``murmur_hash32`` entry
# point in murmur_hash.py does the same device dispatch for callers that import
# it directly.
register_kernel(
    KernelSpec(
        op="sampling.murmur_hash32",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.sampling.murmur_hash:_murmur_hash32_jit",
        capabilities=_CUDA,
        description="MurmurHash3 x86_32 (CUDA JIT kernel).",
    )
)
register_kernel(
    KernelSpec(
        op="sampling.murmur_hash32",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.sampling.murmur_hash:_murmur_hash32_triton",
        capabilities=_HIP,
        description="MurmurHash3 x86_32 (Triton reference, ROCm).",
    )
)
