# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 image generation canvas and the resize back to the requested size.

A request smaller than ``min_generation_area`` pixels is generated on a larger
canvas with the same aspect ratio: both sides scale by
``sqrt(min_generation_area / (width * height))`` and round to the nearest 32px
multiple (one 2x2 patch of 16px VAE cells). A request at or above the area is
only aligned to the 32px grid. After decoding, the canvas image is resized back
to the requested width x height with PIL LANCZOS on the 8-bit image.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import torch
from PIL import Image

MINIMAX_H3_IMAGE_PIXEL_GRID = 32
# 32x32 is the smallest aligned request, so the lower bound never enlarges a
# request; the upper bound is the 4096x4096 tier.
MINIMAX_H3_IMAGE_MIN_GENERATION_AREA_RANGE = (32 * 32, 4096 * 4096)


def minimax_h3_validate_min_generation_area(value: object, path: str) -> int:
    low, high = MINIMAX_H3_IMAGE_MIN_GENERATION_AREA_RANGE
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not low <= value <= high
    ):
        raise ValueError(
            f"{path} must be an integer pixel area within [{low}, {high}], got {value!r}"
        )
    return value


def _nearest_grid_multiple(value: float) -> int:
    # Python round: ties to even.
    grid = MINIMAX_H3_IMAGE_PIXEL_GRID
    return max(grid, int(round(value / grid)) * grid)


def minimax_h3_image_generation_canvas(
    width: int, height: int, min_generation_area: int
) -> tuple[int, int]:
    """The 32px-aligned ``(width, height)`` the DiT generates for a request."""
    scale = max(1.0, sqrt(min_generation_area / (width * height)))
    return _nearest_grid_multiple(width * scale), _nearest_grid_multiple(height * scale)


def minimax_h3_resize_image_output(
    pixels: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    """Resize decoded ``[B, 3, H, W]`` pixels in [0, 1] to ``width x height``.

    Each image is quantized exactly as the output writer does
    (``(x * 255).clamp(0, 255).to(uint8)``), resized with PIL LANCZOS and
    returned as ``uint8 / 255``, which the writer maps back to the same bytes.
    """
    if pixels.ndim != 4 or pixels.shape[1] != 3:
        raise ValueError(
            f"expected decoded [B, 3, H, W] pixels, got {list(pixels.shape)}"
        )
    resized = []
    for image in pixels.cpu():
        array = (image * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        output = Image.fromarray(array).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        resized.append(torch.from_numpy(np.array(output, copy=True)).permute(2, 0, 1))
    return torch.stack(resized).float().div_(255)


__all__ = [
    "MINIMAX_H3_IMAGE_MIN_GENERATION_AREA_RANGE",
    "MINIMAX_H3_IMAGE_PIXEL_GRID",
    "minimax_h3_image_generation_canvas",
    "minimax_h3_resize_image_output",
    "minimax_h3_validate_min_generation_area",
]
