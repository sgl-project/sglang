"""Sampling kernels: softmax, argmax and top-k / top-p probability renormalization."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Union

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
    PlatformInfo,
)

if TYPE_CHECKING:
    import torch


@functools.cache
def _softmax_backend() -> KernelBackend:
    """The softmax kernel this process runs, fixed by its platform."""
    platform = PlatformInfo.detect()
    # TODO: test whether Hopper is ok; now only Blackwell/Rubin
    cuda_sm100 = CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 10))
    if cuda_sm100.is_satisfied_by(platform):
        return KernelBackend.JIT
    if platform.is_hip:
        return KernelBackend.TRITON
    return KernelBackend.TORCH


def softmax(
    logits: torch.Tensor,
    temperatures: Optional[torch.Tensor] = None,
    *,
    rows_per_request: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``softmax(logits / T)`` per row into ``out`` (in place when ``out is logits``).

    ``temperatures`` holds one value per request, each covering ``rows_per_request``
    consecutive rows. Shapes the platform's kernel does not take run on torch.
    """
    import torch

    backend = _softmax_backend()
    if backend is KernelBackend.JIT:
        from sglang.kernels.ops.sampling import softmax_kernel as impl

        if impl.can_softmax(logits, out):
            return impl.softmax(
                logits,
                temperatures,
                rows_per_request=rows_per_request,
                out=out,
            )
    if temperatures is None:
        return torch.softmax(logits, dim=-1, out=out)
    temperatures = temperatures.reshape(-1, 1)
    if rows_per_request > 1:
        temperatures = temperatures.repeat_interleave(rows_per_request, dim=0)
    if backend is KernelBackend.TRITON:
        from sglang.kernels.ops.sampling.temperature_softmax import (
            temperature_softmax,
        )

        probs = temperature_softmax(logits, temperatures)
        return probs if out is None else out.copy_(probs)
    return torch.softmax(logits / temperatures, dim=-1, out=out)


def argmax(
    logits: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Row-wise int64 argmax of ``[rows, vocab]`` into ``out``; ties take the lowest index.

    The JIT kernel never picks a NaN, while the torch fallback (odd layouts) does.
    """
    import torch

    from sglang.kernels.ops.sampling import argmax_kernel as impl

    if impl.can_argmax(logits):
        return impl.argmax(logits, out=out, index_dtype=torch.int64)
    return torch.argmax(logits, dim=-1, out=out)


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


register_kernel(
    KernelSpec(
        op="sampling.argmax",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.sampling.argmax_kernel:argmax",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
        format_signature=FormatSignature(
            supported_dtypes=("float16", "bfloat16", "float32"),
            description="row-wise argmax over [batch, vocab]; ties take the lowest index",
        ),
        description="Split-vocab row-wise argmax for greedy sampling.",
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


__all__ = ["argmax", "softmax", "top_k_renorm_probs", "top_p_renorm_probs"]


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
