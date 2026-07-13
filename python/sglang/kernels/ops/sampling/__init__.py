"""Sampling kernels (top-k / top-p probability renormalization)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import FormatSignature, KernelBackend, KernelSpec

if TYPE_CHECKING:
    import torch

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


__all__ = ["top_k_renorm_probs", "top_p_renorm_probs"]
