# SPDX-License-Identifier: Apache-2.0
"""Lightweight validation shared by H3 metadata preflight and full parsing."""

import math
from collections.abc import Mapping
from typing import Any


def validate_minimax_h3_sigma_shift_scales(
    model_index: Mapping[str, Any],
) -> dict[str, float]:
    path = "model_index.json._minimax_h3"
    if not isinstance(model_index, Mapping):
        raise ValueError("model_index.json must be an object")
    raw = model_index.get("_minimax_h3")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must be an object")
    path += ".sigma_shift_scales"
    scales = raw.get("sigma_shift_scales")
    if not isinstance(scales, Mapping):
        raise ValueError(f"{path} must be an object")
    result = {}
    for modality in ("video", "audio"):
        value = scales.get(modality)
        error = f"{path}.{modality} must be a positive finite number"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(error)
        try:
            scale = float(value)
        except OverflowError as exc:
            raise ValueError(error) from exc
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(error)
        result[modality] = scale
    return result
