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

if TYPE_CHECKING:
    import torch

register_kernel(
    KernelSpec(
        op="sampling.softmax",
        backend=KernelBackend.FLASHINFER,
        target="sglang.kernels.cake_kernels.sampling:softmax",
        capabilities=frozenset(
            {
                CapabilityRequirement.cuda(min_sm=(10, 3), max_sm=(10, 3)),
            }
        ),
        format_signature=FormatSignature(
            supported_dtypes=("float32",),
            description="FP32 [batch, vocab] softmax probabilities",
        ),
        description="Cake Blackwell softmax distributed by FlashInfer.",
    )
)


def cake_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Explicit Cake softmax entry point for Blackwell qualification."""
    return get_kernel("sampling.softmax", KernelBackend.FLASHINFER)(logits)


def softmax(
    logits: torch.Tensor, temperatures: torch.Tensor | None = None
) -> torch.Tensor:
    """Softmax for sampler logits, optionally with per-row temperatures."""
    import torch

    if temperatures is not None and logits.is_cuda and torch.version.hip is not None:
        from sglang.kernels.ops.sampling.temperature_softmax import (
            temperature_softmax,
        )

        return temperature_softmax(logits, temperatures)

    if temperatures is not None:
        logits = logits / temperatures
    from sglang.kernels.cake_kernels.sampling import supports_softmax

    if supports_softmax(logits):
        return cake_softmax(logits)
    return torch.softmax(logits, dim=-1)


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


def _renorm_backend(probs: torch.Tensor) -> str:
    import torch

    # CUDA: the sgl_kernel/FlashInfer op. ROCm: the Triton fallback. MUSA and
    # XPU: their sgl_kernel builds. Else (CPU, declined NPU rows): torch.
    if probs.is_cuda:
        return "triton" if torch.version.hip is not None else "aot"
    if probs.device.type in ("musa", "xpu"):
        return "sgl_kernel"
    return "torch"


def top_k_renorm_probs(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    """Renormalize ``probs`` by top-k thresholding, on any device."""
    backend = _renorm_backend(probs)
    if backend == "aot":
        return get_kernel("sampling.top_k_renorm_probs", KernelBackend.AOT)(
            probs, top_k
        )
    if backend == "triton":
        from sglang.kernels.ops.sampling.renorm_triton import (
            top_k_renorm_probs_triton,
        )

        return top_k_renorm_probs_triton(probs, top_k)
    if backend == "sgl_kernel":
        from sgl_kernel import top_k_renorm_prob

        return top_k_renorm_prob(probs, top_k)
    from sglang.kernels.ops.sampling.renorm_torch import top_k_renorm_probs_torch

    return top_k_renorm_probs_torch(probs, top_k)


def top_p_renorm_probs(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Renormalize ``probs`` by top-p thresholding, on any device."""
    backend = _renorm_backend(probs)
    if backend == "aot":
        return get_kernel("sampling.top_p_renorm_probs", KernelBackend.AOT)(
            probs, top_p
        )
    if backend == "triton":
        from sglang.kernels.ops.sampling.renorm_triton import (
            top_p_renorm_probs_triton,
        )

        return top_p_renorm_probs_triton(probs, top_p)
    if backend == "sgl_kernel":
        from sgl_kernel import top_p_renorm_prob

        return top_p_renorm_prob(probs, top_p)
    from sglang.kernels.ops.sampling.renorm_torch import top_p_renorm_probs_torch

    return top_p_renorm_probs_torch(probs, top_p)


__all__ = ["cake_softmax", "softmax", "top_k_renorm_probs", "top_p_renorm_probs"]


# Migrated from srt/layers/utils/hash.py (RFC #29630, Phase 2.5).
register_kernel(
    KernelSpec(
        op="sampling.murmur_hash32",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.sampling.murmur_hash:murmur_hash32",
    )
)


register_kernel(
    KernelSpec(
        op="sampling.denoiser_statistics",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.sampling.denoiser_statistics:denoiser_statistics",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
        format_signature=FormatSignature(
            supported_dtypes=("float32",),
            description="FP32 probabilities, entropy and raw-logit argmax for contiguous [B,M,V] logits; optional BF16 probabilities.",
        ),
    )
)
