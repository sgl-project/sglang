"""Sampling kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CUDA_ONLY,
    CapabilityRequirement,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

register_kernel(
    KernelSpec(
        op="sampling.softmax",
        backend=KernelBackend.CAKE,
        target="sglang.kernels.cake_kernels.sampling:softmax",
        capabilities=frozenset(
            {
                CapabilityRequirement.cuda(min_sm=(10, 3), max_sm=(10, 3)),
            }
        ),
    )
)


def cake_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Explicit Cake softmax entry point for Blackwell qualification."""
    return get_kernel("sampling.softmax", KernelBackend.CAKE)(logits)


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


__all__ = ["cake_softmax", "softmax"]


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
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="sampling.softcap_row_argmax",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.sampling.softcap_row_argmax:softcap_row_argmax",
        capabilities=CUDA_ONLY,
    )
)
