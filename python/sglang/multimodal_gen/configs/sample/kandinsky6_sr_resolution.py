# SPDX-License-Identifier: Apache-2.0
"""Pure resolution-scale helpers of Kandinsky 6 video super-resolution.

Deliberately kept outside the runtime stage package
(``runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr``): this module is
imported by ``configs/sample/kandinsky6_sr.py`` for request validation, and that sampling
config is itself imported while ``registry`` builds its model table, before
``runtime.pipelines_core.__init__`` (which in turn needs ``registry.get_model_info``) has
finished importing. Importing anything under ``runtime.pipelines_core`` from here would
close that cycle (``registry -> this sampling config -> pipelines_core.__init__ ->
registry.get_model_info``) and break registry-first imports for every model, not just this
one -- see the SGLang PR review comment on ``configs/sample/kandinsky6_sr.py``.

Has no sglang runtime-package dependency: only the stdlib. The runtime stage package's own
``video_utils.py`` imports the same names from here instead of redefining them, so the SR
output stage and the sampling-params validator agree on one definition.
"""

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
    """Isotropic-fit target: shrink ``sr_hw`` to fit inside ``bucket_hw``.

    One scale factor for both axes, so the aspect is preserved exactly: the binding side
    lands on the bucket, the other comes out at (or under) it, snapped to even for the video
    codec. A result already inside the bucket is left alone -- fitting never upscales.
    """
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
    """Resolve a target-resolution spec into the final ``(H, W)``.

    A tier name (``hd`` / ``fullhd`` / ``2k``) resolves against the SR result's aspect ratio,
    picking the closest entry of that tier -- the same closest-aspect rule the tiling uses to
    choose a base resolution. An explicit ``WxH`` is taken as the bucket verbatim.

    ``mode`` then decides how the bucket is honoured: ``"fit"`` (the default) preserves the
    aspect exactly -- an isotropic downscale into the bucket, so an off-tier source is never
    squeezed; ``"exact"`` returns the bucket itself, trading a small anisotropy for exact
    delivery dimensions. ``validate_only`` parses the spec (for request validation) without
    computing a real downscale.

    Args:
        spec: Tier name, ``"WxH"``, or ``None`` / ``"none"`` to keep the raw SR result.
        sr_hw: The SR result's ``(H, W)``, used to pick a tier entry.
        mode: ``"fit"`` (aspect-preserving) or ``"exact"``.
        validate_only: Parse and reject a bad spec without resolving a downscale.

    Returns:
        The target ``(H, W)``, or ``None`` when no resizing is requested (or, in ``fit``
        mode, needed).

    Raises:
        ValueError: On an unknown tier name or a malformed ``WxH``.
    """
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
