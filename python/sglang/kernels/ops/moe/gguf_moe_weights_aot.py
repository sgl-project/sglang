"""Native GGUF route weighting for CUDA and MUSA builds."""

from typing import Callable, Optional

import torch


def apply_gguf_moe_weights(
    outputs: torch.Tensor, weights: torch.Tensor, expert_ids: torch.Tensor
) -> None:
    """Weight rows in place, rounding each product before native MoE reduction."""
    import sgl_kernel  # noqa: F401

    torch.ops.sgl_kernel.apply_gguf_moe_weights.default(outputs, weights, expert_ids)


def resolve_gguf_moe_weights(device_type: str) -> Optional[Callable]:
    """Resolve once at backend import, retaining compatibility with older wheels.

    This native operator ships with negative-ID-safe GGUF vector projections,
    early MMQ slack bounds, and matching Q8_1 value/scale row indexing. Its
    device registration, rather than the Python wrapper's presence, establishes
    that the rebuilt provider can consume raw routes without Python clearing.
    """
    import sgl_kernel  # noqa: F401

    try:
        torch.ops.sgl_kernel.apply_gguf_moe_weights.default
    except AttributeError:
        return None
    key = torch._C._dispatch_key_for_device(device_type)
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "sgl_kernel::apply_gguf_moe_weights", key
    ):
        return None
    return apply_gguf_moe_weights
