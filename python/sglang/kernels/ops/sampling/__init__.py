"""Sampling kernels (top-k / top-p probability renormalization)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    DeviceType,
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


_CPU = CapabilityRequirement(device=DeviceType.CPU)
_RENORM_BACKENDS = (
    # The sgl_kernel wheel ships the op for CUDA only; ROCm takes the Triton
    # port, and CPU plus the NPU rows the vendor path declines take torch.
    (KernelBackend.AOT, "sgl_kernel.sampling:{name}", {CapabilityRequirement.CUDA}),
    (
        KernelBackend.TRITON,
        "sglang.kernels.ops.sampling.renorm_triton:{name}_triton",
        {CapabilityRequirement.HIP},
    ),
    (
        KernelBackend.TORCH,
        "sglang.kernels.ops.sampling.renorm_torch:{name}_torch",
        {_CPU, CapabilityRequirement.NPU},
    ),
)
for _name, _cutoff in (
    ("top_k_renorm_probs", "top-k"),
    ("top_p_renorm_probs", "top-p"),
):
    for _backend, _target, _capabilities in _RENORM_BACKENDS:
        register_kernel(
            KernelSpec(
                op=f"sampling.{_name}",
                backend=_backend,
                target=_target.format(name=_name),
                capabilities=frozenset(_capabilities),
                format_signature=FormatSignature(
                    description=f"renormalize probs by {_cutoff} thresholding; returns tensor"
                ),
                description=f"{_cutoff} probability renormalization ({_backend.value}).",
            )
        )


def _vendor_sgl_kernel(probs: torch.Tensor) -> bool:
    # TODO(RFC #29630): drop once DeviceType grows MUSA / XPU; the platform
    # probe reads them as CPU and would resolve to the torch reference.
    return probs.device.type in ("musa", "xpu")


def _registry_backend(probs: torch.Tensor) -> Optional[KernelBackend]:
    # The registry resolves by platform, so a CPU tensor on a GPU host (unit
    # tests, host-side staging) has to name the torch reference itself.
    return KernelBackend.TORCH if probs.device.type == "cpu" else None


def top_k_renorm_probs(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    """Renormalize ``probs`` by top-k thresholding."""
    if _vendor_sgl_kernel(probs):
        from sgl_kernel import top_k_renorm_prob

        return top_k_renorm_prob(probs, top_k)
    return get_kernel("sampling.top_k_renorm_probs", _registry_backend(probs))(
        probs, top_k
    )


def top_p_renorm_probs(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Renormalize ``probs`` by top-p thresholding."""
    if _vendor_sgl_kernel(probs):
        from sgl_kernel import top_p_renorm_prob

        return top_p_renorm_prob(probs, top_p)
    return get_kernel("sampling.top_p_renorm_probs", _registry_backend(probs))(
        probs, top_p
    )


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
