# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def prune_fully_masked_cameras(
    camera_order: tuple[str, ...],
    image_masks: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    kept = tuple(
        name
        for name in camera_order
        if image_masks[name].device.type != "cpu"
        or bool(image_masks[name].any().item())
    )
    # The vision encoder requires at least one camera. Keep its existing
    # all-masked behavior rather than introducing a new text-only path.
    return kept or camera_order
