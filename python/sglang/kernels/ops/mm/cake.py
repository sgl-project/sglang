"""Cake (FlashInfer) backends for the ``mm`` operator group: Kimi-K3 vision tower.

Metadata-only at import time: ``KernelSpec`` targets point at the lazy
adapters in :mod:`sglang.kernels.cake_kernels.mm`, which import FlashInfer only
when a kernel is actually called.

The vision tower is multimodal encoding (MoonViT-3D + PatchMergerV2 over patch
pixels), so it is classified under ``mm`` with the other multimodal
input-processing kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

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

_SM100_103 = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 3))})

register_kernel(
    KernelSpec(
        op="mm.kimi_k3_vision_tower",
        backend=KernelBackend.FLASHINFER,
        target="sglang.kernels.cake_kernels.mm:kimi_k3_vision_tower",
        capabilities=_SM100_103,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description=(
                "BF16 pixel_values [T,3,14,14] + grid_thws + weights -> out [N,7168]; "
                "one-shot (prepares dict weights and the host plan per call)"
            ),
        ),
        description=(
            "Cake Kimi-K3 vision tower (MoonViT-3D + PatchMergerV2), one-shot. "
            "Distributed by FlashInfer."
        ),
    )
)

register_kernel(
    KernelSpec(
        op="mm.prepare_kimi_k3_vision_tower",
        backend=KernelBackend.FLASHINFER,
        target="sglang.kernels.cake_kernels.mm:prepare_kimi_k3_vision_tower",
        capabilities=_SM100_103,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            in_place=True,
            description=(
                "prepare -> runner.launch() writes out [N,7168] with no allocation; "
                "bound to one grid_thws / tensor-binding set"
            ),
        ),
        description=(
            "Cake Kimi-K3 vision tower prepared runner (CUDA-graph capturable). "
            "Distributed by FlashInfer."
        ),
    )
)

register_kernel(
    KernelSpec(
        op="mm.prepare_kimi_k3_vision_weights",
        backend=KernelBackend.FLASHINFER,
        target="sglang.kernels.cake_kernels.mm:prepare_kimi_k3_vision_weights",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description=(
                "dict of BF16 nn.Linear-style weights -> PreparedWeights "
                "(patch-proj padding, contiguous copies); once per model"
            ),
        ),
        description=(
            "Cake Kimi-K3 vision tower weight preparation. Distributed by FlashInfer."
        ),
    )
)


def cake_kimi_k3_vision_tower(
    pixel_values: torch.Tensor,
    grid_thws: Sequence[Sequence[int]],
    weights: Any,
    out: Optional[torch.Tensor] = None,
    *,
    plan: Any = None,
    pos_rows: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return get_kernel("mm.kimi_k3_vision_tower", KernelBackend.FLASHINFER)(
        pixel_values, grid_thws, weights, out, plan=plan, pos_rows=pos_rows
    )


def cake_prepare_kimi_k3_vision_tower(
    pixel_values: torch.Tensor,
    grid_thws: Sequence[Sequence[int]],
    weights: Any,
    out: Optional[torch.Tensor] = None,
    *,
    plan: Any = None,
    pos_rows: Optional[torch.Tensor] = None,
) -> Any:
    """Explicit Cake entry point; returns the FlashInfer ``KimiK3VisionTowerRunner``."""
    return get_kernel("mm.prepare_kimi_k3_vision_tower", KernelBackend.FLASHINFER)(
        pixel_values, grid_thws, weights, out, plan=plan, pos_rows=pos_rows
    )


def cake_prepare_kimi_k3_vision_weights(weights: dict) -> Any:
    """Explicit Cake entry point; returns the FlashInfer ``PreparedWeights``."""
    return get_kernel("mm.prepare_kimi_k3_vision_weights", KernelBackend.FLASHINFER)(
        weights
    )


__all__ = [
    "cake_kimi_k3_vision_tower",
    "cake_prepare_kimi_k3_vision_tower",
    "cake_prepare_kimi_k3_vision_weights",
]
