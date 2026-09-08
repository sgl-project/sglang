"""Resolution and post-decode geometry helpers for HunyuanImage-3."""

import math

OUTPUT_GEOMETRY_EXTRA_KEY = "hunyuan_image3_output_geometry"


def build_hunyuan_image3_output_geometry(
    width: int,
    height: int,
    *,
    size_mode: str = "aspect_ratio",
    strategy: str = "native_crop",
    ratio_policy: str = "exact",
    crop_anchor: tuple[float, float] = (0.5, 0.5),
    max_ratio_error: float = 0.0005,
    pad_value: float = 0.0,
) -> dict[str, object]:
    """Create a request-local output contract before native bucketing.

    ``width`` and ``height`` are the user-visible target geometry (or the
    unmodified reference geometry for an edit with no explicit size), not the
    processor-selected bucket.  Keeping this dictionary on ``Req.extra`` makes
    it safe to use with cloned requests and decoder disaggregation.
    """
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("HunyuanImage-3 target dimensions must be positive")
    if size_mode not in {"aspect_ratio", "exact_size"}:
        raise ValueError("size_mode must be 'aspect_ratio' or 'exact_size'")
    if strategy not in {"native_crop", "native_pad"}:
        raise ValueError("strategy must be 'native_crop' or 'native_pad'")
    if ratio_policy not in {"exact", "approximate"}:
        raise ValueError("ratio_policy must be 'exact' or 'approximate'")
    if len(crop_anchor) != 2 or any(
        not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in crop_anchor
    ):
        raise ValueError("crop_anchor must contain two finite values in [0, 1]")
    if not math.isfinite(max_ratio_error) or not 0.0 <= max_ratio_error < 1.0:
        raise ValueError("max_ratio_error must be a finite value in [0, 1)")
    if not math.isfinite(pad_value) or not 0.0 <= pad_value <= 1.0:
        raise ValueError("pad_value must be a finite value in [0, 1]")
    if size_mode == "exact_size" and ratio_policy == "approximate":
        raise ValueError(
            "exact_size requires ratio_policy='exact'; approximate cropping "
            "would require non-uniform resampling to reach the requested size"
        )

    divisor = math.gcd(width, height)
    return {
        "requested_size": [width, height],
        "requested_aspect_ratio": [width // divisor, height // divisor],
        "size_mode": size_mode,
        "strategy": strategy,
        "ratio_policy": ratio_policy,
        "crop_anchor": [float(crop_anchor[0]), float(crop_anchor[1])],
        "max_ratio_error": float(max_ratio_error),
        "pad_value": float(pad_value),
    }


def resolve_hunyuan_image3_output_resolution(
    width: int,
    height: int,
    explicit_fields: set[str],
    reference_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Return the raw target size used for native aspect-ratio bucketing.

    Explicit dimensions take precedence. Image editing without an explicit
    size inherits the unmodified reference size instead of the generic
    1280x720 pipeline default.
    """
    if reference_size is not None and not {"width", "height"} & explicit_fields:
        width, height = reference_size
    return int(width), int(height)
