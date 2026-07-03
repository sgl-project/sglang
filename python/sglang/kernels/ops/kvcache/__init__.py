"""KV-cache write/transfer kernels.

This group wraps the Triton ``reshape_and_cache`` launcher that currently lives
under ``sglang.srt.layers.attention.triton_ops``. Only a thin wrapper is added
here; physically moving the Triton source into this package is deferred to a
later phase (RFC #29630) to keep this change low-risk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import FormatSignature, KernelBackend, KernelSpec

if TYPE_CHECKING:
    import torch

register_kernel(
    KernelSpec(
        op="kvcache.reshape_and_cache_flash",
        backend=KernelBackend.TRITON,
        target="sglang.srt.layers.attention.triton_ops.cache_ops:launch_reshape_and_cache_flash",
        priority=10,
        format_signature=FormatSignature(
            in_place=True,
            description="write token-major K/V into paged KV cache layout",
        ),
        description="Reshape-and-cache (Triton launcher).",
    )
)


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    swa_slot_mapping: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> None:
    """Write token-major ``key``/``value`` into paged KV cache layout."""
    return get_kernel("kvcache.reshape_and_cache_flash", KernelBackend.TRITON)(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        swa_slot_mapping,
        k_scale,
        v_scale,
    )


__all__ = ["reshape_and_cache_flash"]
