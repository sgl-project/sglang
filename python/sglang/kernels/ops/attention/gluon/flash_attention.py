# SPDX-License-Identifier: Apache-2.0
"""Version-neutral entry point for Gluon FlashAttention.

Implementation selection belongs here. Each version/architecture package owns
its dependency checks, launch policy, and kernels; importing this module does
not import Gluon or query a GPU.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch


@functools.lru_cache(maxsize=None)
def _load_implementation(
    architecture: str,
) -> tuple[Callable[..., torch.Tensor] | None, str | None]:
    if architecture == "gfx1250":
        from sglang.kernels.ops.attention.gluon.fa3.gfx1250 import load_forward

        return load_forward()
    return None, f"GPU architecture {architecture or 'unknown'}"


def get_flash_attention_impl(
    device: torch.device,
) -> tuple[Callable[..., torch.Tensor] | None, str | None]:
    """Return the implementation for this device, or its unavailability reason."""
    if device.type != "cuda" or torch.version.hip is None:
        return None, "Gluon FlashAttention requires a ROCm device"
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "").split(":")[0]
    return _load_implementation(architecture)


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run dense, noncausal BSHD attention using the device's Gluon implementation.

    The selected kernel validates its tensor contract. Unsupported devices or
    missing compiler APIs raise here; application backends own fallback policy.
    Launch overrides remain local to each implementation.
    """
    implementation, reason = get_flash_attention_impl(query.device)
    if implementation is None:
        raise RuntimeError(f"Gluon FlashAttention unavailable: {reason}")
    return implementation(query, key, value, softmax_scale=softmax_scale)


__all__ = ["flash_attention", "get_flash_attention_impl"]
