"""Sampling kernels."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CUDA_ONLY,
    CapabilityRequirement,
    KernelBackend,
    KernelSpec,
    PlatformInfo,
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
                CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 0)),
                CapabilityRequirement.cuda(min_sm=(10, 3), max_sm=(10, 3)),
            }
        ),
    )
)


def cake_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Explicit Cake softmax entry point for Blackwell qualification."""
    return get_kernel("sampling.softmax", KernelBackend.CAKE)(logits)


# The JIT softmax's config model is fit on the SM100 family only.
_JIT_SOFTMAX_CAPABILITY = CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 10))

register_kernel(
    KernelSpec(
        op="sampling.softmax",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.sampling.softmax_jit:softmax",
        capabilities=frozenset({_JIT_SOFTMAX_CAPABILITY}),
    )
)
register_kernel(
    KernelSpec(
        op="sampling.argmax",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.sampling.argmax_jit:argmax",
        capabilities=CUDA_ONLY,
    )
)


@functools.cache
def _use_jit_softmax() -> bool:
    return _JIT_SOFTMAX_CAPABILITY.is_satisfied_by(PlatformInfo.detect())


def softmax(
    logits: torch.Tensor,
    temperatures: Optional[torch.Tensor] = None,
    *,
    rows_per_request: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``softmax(logits / T)`` per row in fp32, into ``out`` (in place when ``out is logits``).

    ``temperatures`` holds one value per request, each covering ``rows_per_request``
    consecutive rows.
    """
    import torch

    if logits.is_cuda and _use_jit_softmax():
        from sglang.kernels.ops.sampling import softmax_jit

        if softmax_jit.can_softmax(logits, out):
            return softmax_jit.softmax(
                logits, temperatures, rows_per_request=rows_per_request, out=out
            )
    if temperatures is not None and rows_per_request > 1:
        temperatures = temperatures.repeat_interleave(rows_per_request, dim=0)
    if temperatures is not None and logits.is_cuda and torch.version.hip is not None:
        from sglang.kernels.ops.sampling.softmax_triton import (
            temperature_softmax,
        )

        probs = temperature_softmax(logits, temperatures)
    else:
        if temperatures is not None:
            logits = logits / temperatures
        from sglang.kernels.cake_kernels.sampling import supports_softmax

        if supports_softmax(logits):
            probs = cake_softmax(logits)
        else:
            probs = torch.softmax(logits, dim=-1)
    return probs if out is None else out.copy_(probs)


def argmax(logits: torch.Tensor) -> torch.Tensor:
    """Row-wise int64 argmax of ``[rows, vocab]``; a value tie takes the lowest index.

    The JIT kernel never picks a NaN, while the torch fallback (unaligned rows) does.
    """
    import torch

    from sglang.kernels.ops.sampling import argmax_jit

    if logits.is_cuda and argmax_jit.can_argmax(logits):
        return argmax_jit.argmax(logits, index_dtype=torch.int64)
    return torch.argmax(logits, dim=-1)


__all__ = ["argmax", "cake_softmax", "softmax"]


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
