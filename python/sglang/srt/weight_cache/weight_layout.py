# SPDX-License-Identifier: Apache-2.0
"""Processed weight layout settings shared by presharded and snapshot loaders."""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def flashinfer_cutedsl_swap_ab(target_device: torch.device | None) -> bool | None:
    """Return the row layout selected by the active CuTeDSL MoE path, if used."""
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend

    a2a_backend = get_moe_a2a_backend()
    if not (
        get_moe_runner_backend().is_flashinfer_cutedsl()
        and (a2a_backend.is_none() or a2a_backend.is_flashinfer())
    ):
        return None

    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
        use_cutedsl_moe_swap_ab,
    )

    if target_device is None:
        import torch

        from sglang.srt.platforms import current_platform

        target_device = torch.device(current_platform.device_type)
    # Both layouts have the same shape, but interleave gate and up rows differently.
    return use_cutedsl_moe_swap_ab(target_device)


def normalize_init_expert_location(value: str | None) -> str | None:
    """Include the contents of a file-backed initial expert map in the cache key."""
    if value is None or value == "trivial":
        return value
    if value.endswith((".json", ".pt")) and os.path.isfile(value):
        digest = hashlib.sha1()
        with open(value, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"file:{os.path.basename(value)}:sha1:{digest.hexdigest()[:16]}"
    return value
