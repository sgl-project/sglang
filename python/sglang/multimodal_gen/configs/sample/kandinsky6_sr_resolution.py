# SPDX-License-Identifier: Apache-2.0
"""SR resolution validation shared by sampling parameters and output stages.

Keep this module independent of runtime to avoid the registry import cycle."""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

# from-video total scales (2.25 = x1.125 pixel pre-upscale + x2 tiling).
SUPPORTED_RESOLUTION_SCALES: tuple[float, ...] = (2, 4, 2.25)

# Delivery tiers for the target resolution: (H, W) candidates per tier -- landscape 16:9,
# square, portrait 9:16; the closest-aspect entry wins.
TARGET_RESOLUTIONS: dict[str, tuple[tuple[int, int], ...]] = {
    "hd": ((720, 1280), (720, 720), (1280, 720)),
    "fullhd": ((1080, 1920), (1080, 1080), (1920, 1080)),
    "2k": ((1440, 2560), (1440, 1440), (2560, 1440)),
}

TargetResizeMode = Literal["fit", "exact"]

# Log a warning when the SR and target aspect ratios differ by more than this (relative) --
# a mild anisotropic squeeze is fine, a big one means the wrong tier for the route.
ASPECT_MISMATCH_TOLERANCE = 0.02


def fit_within(
    sr_hw: tuple[int, int], bucket_hw: tuple[int, int]
) -> tuple[int, int] | None:
    """Fit within the bucket without upscaling; preserve aspect and codec-even sizes."""
    scale = min(bucket_hw[0] / sr_hw[0], bucket_hw[1] / sr_hw[1])
    if scale >= 1.0:
        logger.warning(
            "SR result %sx%s already fits inside the %sx%s bucket -- nothing to "
            "downscale, keeping it as is.",
            sr_hw[1],
            sr_hw[0],
            bucket_hw[1],
            bucket_hw[0],
        )
        return None
    height, width = (
        min(bucket, round(side * scale / 2) * 2)
        for bucket, side in zip(bucket_hw, sr_hw, strict=True)
    )
    return height, width


def resolve_target_hw(
    spec: str | None,
    sr_hw: tuple[int, int],
    mode: TargetResizeMode = "fit",
    validate_only: bool = False,
) -> tuple[int, int] | None:
    """Resolve a tier or WxH bucket to (H, W), or None for no resize.

    fit preserves aspect; exact returns the bucket dimensions. validate_only
    checks syntax without needing an actual SR result."""
    if mode not in ("fit", "exact"):
        raise ValueError(
            f"sr_target_resize_mode must be 'fit' or 'exact', got {mode!r}"
        )
    if spec is None or spec.lower() == "none":
        return None
    key = spec.lower()
    if key in TARGET_RESOLUTIONS:
        source_ratio = sr_hw[1] / sr_hw[0]
        bucket = min(
            TARGET_RESOLUTIONS[key], key=lambda hw: abs(hw[1] / hw[0] - source_ratio)
        )
    else:
        parts = key.split("x")
        if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
            raise ValueError(
                f"sr_target_resolution must be one of {sorted(TARGET_RESOLUTIONS)} or "
                f"WxH (e.g. 1280x720), got {spec!r}"
            )
        width, height = (int(part) for part in parts)
        bucket = (height, width)
    if mode == "exact" or validate_only:
        return bucket
    return fit_within(sr_hw, bucket)


__all__ = [
    "ASPECT_MISMATCH_TOLERANCE",
    "SUPPORTED_RESOLUTION_SCALES",
    "TARGET_RESOLUTIONS",
    "TargetResizeMode",
    "fit_within",
    "resolve_target_hw",
]
