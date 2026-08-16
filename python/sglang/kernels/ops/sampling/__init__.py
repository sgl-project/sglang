"""Sampling kernels (top-k / top-p probability renormalization)."""

from __future__ import annotations

from typing import Union

import torch

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    DeviceType,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

register_kernel(
    KernelSpec(
        op="sampling.top_k_renorm_probs",
        backend=KernelBackend.AOT,
        target="sgl_kernel.sampling:top_k_renorm_probs",
        capabilities=frozenset((CapabilityRequirement.CUDA,)),
        format_signature=FormatSignature(
            description="renormalize probs by top-k thresholding; returns tensor"
        ),
        description="Top-k probability renormalization (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="sampling.top_k_renorm_probs",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.sampling.renorm_triton:top_k_renorm_probs_triton",
        capabilities=frozenset((CapabilityRequirement.HIP,)),
        format_signature=FormatSignature(
            supported_dtypes=("float32",),
            description="threshold-based top-k probability renormalization",
        ),
    )
)
register_kernel(
    KernelSpec(
        op="sampling.top_k_renorm_probs",
        backend=KernelBackend.TORCH,
        target="sglang.kernels.ops.sampling.renorm:top_k_renorm_probs_torch",
        capabilities=frozenset(
            (
                CapabilityRequirement.HIP,
                CapabilityRequirement.NPU,
                CapabilityRequirement(device=DeviceType.CPU),
            )
        ),
        format_signature=FormatSignature(
            supported_dtypes=("float32",),
            description="portable threshold-based top-k probability renormalization",
        ),
    )
)
register_kernel(
    KernelSpec(
        op="sampling.top_p_renorm_probs",
        backend=KernelBackend.AOT,
        target="sgl_kernel.sampling:top_p_renorm_probs",
        capabilities=frozenset((CapabilityRequirement.CUDA,)),
        format_signature=FormatSignature(
            description="renormalize probs by top-p thresholding; returns tensor"
        ),
        description="Top-p probability renormalization (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="sampling.top_p_renorm_probs",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.sampling.renorm_triton:top_p_renorm_probs_triton",
        capabilities=frozenset((CapabilityRequirement.HIP,)),
        format_signature=FormatSignature(
            supported_dtypes=("float32",),
            description="threshold-based nucleus probability renormalization",
        ),
    )
)
register_kernel(
    KernelSpec(
        op="sampling.top_p_renorm_probs",
        backend=KernelBackend.TORCH,
        target="sglang.kernels.ops.sampling.renorm:top_p_renorm_probs_torch",
        capabilities=frozenset(
            (
                CapabilityRequirement.HIP,
                CapabilityRequirement.NPU,
                CapabilityRequirement(device=DeviceType.CPU),
            )
        ),
        format_signature=FormatSignature(
            supported_dtypes=("float32",),
            description="portable nucleus probability renormalization",
        ),
    )
)


def _renorm_backend(probs: torch.Tensor) -> KernelBackend:
    if probs.device.type == "cuda" and torch.version.hip is None:
        return KernelBackend.AOT
    if probs.device.type == "musa":
        return KernelBackend.AOT
    if probs.device.type == "cuda":
        return KernelBackend.TRITON
    return KernelBackend.TORCH


def top_k_renorm_probs(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    """Renormalize ``probs`` by top-k thresholding."""
    return get_kernel("sampling.top_k_renorm_probs", _renorm_backend(probs))(
        probs, top_k
    )


def top_p_renorm_probs(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Renormalize ``probs`` by top-p thresholding."""
    return get_kernel("sampling.top_p_renorm_probs", _renorm_backend(probs))(
        probs, top_p
    )


__all__ = ["top_k_renorm_probs", "top_p_renorm_probs"]


# Migrated from srt/layers/utils/hash.py (RFC #29630, Phase 2.5).
register_kernel(
    KernelSpec(
        op="sampling.murmur_hash32",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.sampling.murmur_hash:murmur_hash32",
    )
)
