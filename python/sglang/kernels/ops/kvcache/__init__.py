"""KV-cache write/transfer kernels.

This group wraps the Triton ``reshape_and_cache`` launcher, whose implementation
now lives in this package (``sglang.kernels.ops.kvcache.cache_ops``) after being
migrated out of ``sglang.srt.layers.attention.triton_ops`` (RFC #29630).
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
        target="sglang.kernels.ops.kvcache.cache_ops:launch_reshape_and_cache_flash",
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
