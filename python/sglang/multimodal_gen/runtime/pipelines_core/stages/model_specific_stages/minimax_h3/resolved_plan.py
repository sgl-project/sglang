# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 ResolvedPlan: the data-only per-request execution plan.

`minimax_h3_resolve_plan` turns a validated canonical request (see
request_validation.py) into the data-only plan consumed by stages 1-8.
Stages never branch on task names; skips must be explicit in the plan.

Scope notes (adapt_shape_v1):
- all target and material-derived ratios use the single adaptive spatial
  resolver exported by this module. It starts from a 768px nominal short edge,
  applies the 768x1344 soft area cap, then rounds both axes independently to
  the nearest 32px grid.
- ``auto`` uses the task profile: t2va/ref2va resolve to the 16:9 policy
  default, while fl2va defers geometry until material probe facts are
  available. Consumers must fail fast if required evidence is missing.
- per-modality request overrides and task defaults are retained separately so
  the timestep stage can apply request > model config > task default priority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import msgspec

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SUPPORTED_FPS,
    warn_unverified_short_edge,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,
    minimax_h3_task_profile,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_align_frame_count,
    minimax_h3_audio_latent_t,
    minimax_h3_video_latent_t,
)

MINIMAX_H3_SHAPE_POLICY_VERSION = "adapt_shape_v1"
MINIMAX_H3_BASE_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = MINIMAX_H3_BASE_SHORT_EDGE * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_ASPECT_RATIO = 1.0 / 4.0
MINIMAX_H3_MAX_ASPECT_RATIO = 4.0


class MiniMaxH3MaterialPlanItem(msgspec.Struct, frozen=True):
    condition_index: int
    role: str
    condition_type: str
    uri: str
    material_chain: str
    # Request-level semantic frame index.  -1 remains the last-frame sentinel.
    frame_index: int | None = None
    # Concrete pixel-frame index after target 17n+5 alignment.
    resolved_frame_index: int | None = None
    # Per-reference seek applied identically to the visual and audio streams.
    start_time_seconds: float = 0.0


class MiniMaxH3ResolvedPlan(msgspec.Struct, frozen=True):
    task: str
    prompt: str
    seed: int | None
    materials: tuple[MiniMaxH3MaterialPlanItem, ...]
    encoders: dict
    branches: tuple[dict, ...]
    default_flow_shift: float
    default_audio_flow_shift: float
    flow_shift: float | None
    audio_flow_shift: float | None
    shape: dict
    condition_mask: dict


def _parse_aspect_ratio(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"target.aspect_ratio must be 'W:H' or 'auto', got {value!r}")
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"target.aspect_ratio must be integer 'W:H', got {value!r}"
        ) from exc
    if w <= 0 or h <= 0:
        raise ValueError(
            f"target.aspect_ratio components must be positive, got {value!r}"
        )
    return w, h


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _validate_base_short_edge(value: Any) -> int:
    try:
        short_edge = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"target.short_edge must be an integer, got {value!r}"
        ) from exc
    if short_edge != value or short_edge <= 0:
        raise ValueError(f"target.short_edge must be a positive integer, got {value!r}")
    if short_edge != MINIMAX_H3_BASE_SHORT_EDGE:
        warn_unverified_short_edge(short_edge)
    return short_edge


def minimax_h3_resolve_spatial_shape(
    *,
    width: int | float,
    height: int | float,
    base_short_edge: int = MINIMAX_H3_BASE_SHORT_EDGE,
) -> dict[str, Any]:
    """Resolve one display ratio with the ``adapt_shape_v1`` math.

    This is the only implementation of adaptive target geometry. Callers may
    pass an explicit aspect-ratio pair or probed display dimensions; only the
    ratio is significant. The supported ratio range is inclusive 1:4 to 4:1.
    The returned dimensions are always 32px aligned; nearest-grid rounding may
    leave the final area slightly above the pre-round soft pixel budget.
    """
    base_short_edge = _validate_base_short_edge(base_short_edge)
    try:
        source_width = float(width)
        source_height = float(height)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "shape width and height must be positive finite numbers"
        ) from exc
    if (
        not math.isfinite(source_width)
        or not math.isfinite(source_height)
        or source_width <= 0.0
        or source_height <= 0.0
    ):
        raise ValueError("shape width and height must be positive finite numbers")

    ratio = source_width / source_height
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("shape ratio must be a positive finite number")
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(
            "adapt_shape_v1 ratio must be within the inclusive range "
            f"1:4 to 4:1, got {source_width:g}:{source_height:g}"
        )

    if ratio >= 1.0:
        nominal_width = float(base_short_edge) * ratio
        nominal_height = float(base_short_edge)
    else:
        nominal_width = float(base_short_edge)
        nominal_height = float(base_short_edge) / ratio
    nominal_area = nominal_width * nominal_height
    if nominal_area > MINIMAX_H3_MAX_PIXELS:
        size_mode = "area"
        scale = math.sqrt(float(MINIMAX_H3_MAX_PIXELS) / nominal_area)
        nominal_width *= scale
        nominal_height *= scale
    else:
        size_mode = "short_edge"

    resolved_width = _nearest_multiple(nominal_width, MINIMAX_H3_CANVAS_MULTIPLE)
    resolved_height = _nearest_multiple(nominal_height, MINIMAX_H3_CANVAS_MULTIPLE)

    return {
        "geometry": "resolved_v2",
        "shape_policy_version": MINIMAX_H3_SHAPE_POLICY_VERSION,
        "base_short_edge": base_short_edge,
        "effective_short_edge": min(resolved_width, resolved_height),
        "size_mode": size_mode,
        "max_pixels": MINIMAX_H3_MAX_PIXELS,
        "multiple": MINIMAX_H3_CANVAS_MULTIPLE,
        "rounding": "nearest",
        "width": resolved_width,
        "height": resolved_height,
    }


def _resolve_shape(
    target: Mapping[str, Any],
    *,
    geometry_source: str,
    auto_aspect_ratio: str | None = None,
    auto_geometry_source: str | None = None,
) -> dict[str, Any]:
    fps = MINIMAX_H3_SUPPORTED_FPS
    if "duration_seconds" not in target:
        # ref2va duration_from_audio_reference: temporal shape resolves at
        # material time from the reference audio probe. Validation
        # guarantees an audio condition exists.
        shape: dict[str, Any] = {
            "fps": fps,
            "temporal": "deferred_from_audio_reference",
            "geometry_source": geometry_source,
        }
        return _resolve_spatial(
            shape,
            target,
            auto_aspect_ratio=auto_aspect_ratio,
            auto_geometry_source=auto_geometry_source,
        )
    frame_count = minimax_h3_align_frame_count(
        int(round(float(target["duration_seconds"]) * fps))
    )
    duration_seconds = frame_count / fps
    shape = {
        "fps": fps,
        "frame_count": frame_count,
        "video_latent_t": minimax_h3_video_latent_t(frame_count),
        "audio_latent_t": minimax_h3_audio_latent_t(duration_seconds),
        "geometry_source": geometry_source,
    }
    return _resolve_spatial(
        shape,
        target,
        auto_aspect_ratio=auto_aspect_ratio,
        auto_geometry_source=auto_geometry_source,
    )


def _resolve_spatial(
    shape: dict[str, Any],
    target: Mapping[str, Any],
    *,
    auto_aspect_ratio: str | None,
    auto_geometry_source: str | None,
) -> dict[str, Any]:
    aspect_ratio = str(target["aspect_ratio"])
    base_short_edge = _validate_base_short_edge(target.get("short_edge"))
    if aspect_ratio == "auto":
        if auto_aspect_ratio is None:
            # Deferred: canvas comes from material/model geometry at prepare time.
            shape["geometry"] = "deferred"
            shape["geometry_source"] = auto_geometry_source or shape["geometry_source"]
            shape["shape_policy_version"] = MINIMAX_H3_SHAPE_POLICY_VERSION
            shape["base_short_edge"] = base_short_edge
            shape["size_mode"] = "deferred"
            return shape
        aspect_ratio = auto_aspect_ratio
        shape["geometry_source"] = auto_geometry_source or "policy_default"
    ar_w, ar_h = _parse_aspect_ratio(aspect_ratio)
    shape.update(
        minimax_h3_resolve_spatial_shape(
            width=ar_w,
            height=ar_h,
            base_short_edge=base_short_edge,
        )
    )
    return shape


def minimax_h3_resolve_plan(canonical: Mapping[str, Any]) -> MiniMaxH3ResolvedPlan:
    """Canonical request (already validated) -> ResolvedPlan."""
    if not isinstance(canonical, Mapping):
        raise ValueError("canonical request must be a mapping")
    allowed_keys = {
        "schema",
        "task",
        "prompt",
        "conditions",
        "target",
        "seed",
        "flow_shift",
        "audio_flow_shift",
    }
    unknown = set(canonical) - allowed_keys
    if unknown:
        raise ValueError(f"canonical request has unknown fields: {sorted(unknown)}")
    for key in ("schema", "task", "prompt", "conditions", "target"):
        if key not in canonical:
            raise ValueError(f"canonical request missing {key!r}")
    profile = minimax_h3_task_profile(str(canonical["task"]))
    conditions = canonical["conditions"]
    keyframe_conditions = (
        [
            condition
            for condition in conditions
            if isinstance(condition, Mapping) and condition.get("role") == "keyframe"
        ]
        if isinstance(conditions, (list, tuple))
        else []
    )
    if profile.task == "fl2va" or keyframe_conditions:
        signatures = (
            [
                (
                    condition.get("type"),
                    condition.get("role"),
                    condition.get("frame_index"),
                )
                for condition in keyframe_conditions
            ]
            if all(isinstance(condition, Mapping) for condition in keyframe_conditions)
            else []
        )
        frame_signature = tuple(signature[2] for signature in signatures)
        if (
            not signatures
            or any(signature[:2] != ("image", "keyframe") for signature in signatures)
            or frame_signature not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES
        ):
            raise ValueError(
                f"{profile.task} ResolvedPlan requires one or two ordered "
                "image/keyframe "
                "conditions with frame_index [0], [-1], or [0, -1], got "
                f"{signatures!r}"
            )
    shape = _resolve_shape(
        canonical["target"],
        geometry_source=profile.geometry_source,
        auto_aspect_ratio=profile.auto_aspect_ratio,
        auto_geometry_source=profile.auto_geometry_source,
    )

    materials: list[MiniMaxH3MaterialPlanItem] = []
    visual_encode: list[int] = []
    audio_encode: list[int] = []
    keyframe_semantic_indices: list[int] = []
    keyframe_pixel_indices: list[int] = []
    seen_keyframe_pixel_indices: dict[int, int] = {}
    for index, cond in enumerate(canonical["conditions"]):
        rule = profile.rule_for(
            role=str(cond["role"]), condition_type=str(cond["type"])
        )
        frame_index = cond.get("frame_index")
        resolved_frame_index = None
        if rule.requires_frame_index:
            if frame_index is None:
                raise ValueError(f"conditions[{index}].frame_index is required")
            semantic_frame_index = int(frame_index)
            frame_count = int(shape["frame_count"])
            if semantic_frame_index == -1:
                resolved_frame_index = frame_count - 1
            elif 0 <= semantic_frame_index < frame_count:
                resolved_frame_index = semantic_frame_index
            else:
                raise ValueError(
                    f"conditions[{index}].frame_index must be -1 or in "
                    f"[0, {frame_count}) after 17n+5 frame alignment, got "
                    f"{semantic_frame_index}"
                )
            previous = seen_keyframe_pixel_indices.get(resolved_frame_index)
            if previous is not None:
                raise ValueError(
                    f"conditions[{index}].frame_index resolves to "
                    f"{resolved_frame_index}, already bound by "
                    f"conditions[{previous}]"
                )
            seen_keyframe_pixel_indices[resolved_frame_index] = index
            keyframe_semantic_indices.append(semantic_frame_index)
            keyframe_pixel_indices.append(resolved_frame_index)
        materials.append(
            MiniMaxH3MaterialPlanItem(
                condition_index=index,
                role=str(cond["role"]),
                condition_type=str(cond["type"]),
                uri=str(cond["uri"]),
                material_chain=rule.material_chain,
                frame_index=frame_index,
                resolved_frame_index=resolved_frame_index,
                start_time_seconds=float(cond.get("start_time_seconds", 0.0)),
            )
        )
        if rule.visual_tokenizer_encode:
            visual_encode.append(index)
        if rule.audio_tokenizer_encode:
            audio_encode.append(index)

    qwen_condition_indices = [
        index
        for index, condition in enumerate(canonical["conditions"])
        if profile.task != "ref2va" or condition["role"] == "reference"
    ]
    encoders = {
        "qwen": {
            "prompt": canonical["prompt"],
            "ordered_condition_indices": qwen_condition_indices,
        },
        "visual": visual_encode,
        "audio": audio_encode,
    }

    condition_mask: dict[str, Any] = {}
    if keyframe_pixel_indices:
        condition_mask = {
            # Both arrays are request-ordered.  Semantic indices feed Qwen and
            # the RoPE rule; resolved
            # indices are concrete output frames.
            "semantic_frame_indices": keyframe_semantic_indices,
            "pixel_frame_indices": keyframe_pixel_indices,
        }

    return MiniMaxH3ResolvedPlan(
        task=profile.task,
        prompt=str(canonical["prompt"]),
        seed=canonical.get("seed"),
        materials=tuple(materials),
        encoders=encoders,
        branches=profile.branches,
        default_flow_shift=float(profile.default_flow_shift),
        default_audio_flow_shift=float(profile.default_audio_flow_shift),
        flow_shift=(
            float(canonical["flow_shift"])
            if canonical.get("flow_shift") is not None
            else None
        ),
        audio_flow_shift=(
            float(canonical["audio_flow_shift"])
            if canonical.get("audio_flow_shift") is not None
            else None
        ),
        shape=shape,
        condition_mask=condition_mask,
    )


MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY = "minimax_h3_canonical_request"
MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY = "minimax_h3_resolved_plan"


def minimax_h3_plan_from_batch(batch: Any) -> MiniMaxH3ResolvedPlan | None:
    """Resolve (once) and cache the plan for a Req carrying a canonical request.

    Returns None when the request predates the canonical schema (such
    requests keep their existing behavior).
    """
    extra = getattr(batch, "extra", None)
    if not isinstance(extra, Mapping):
        return None
    cached = extra.get(MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY)
    if cached is not None:
        if not isinstance(cached, MiniMaxH3ResolvedPlan):
            raise ValueError(
                f"batch.extra[{MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY!r}] must be a "
                "MiniMaxH3ResolvedPlan"
            )
    canonical = extra.get(MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY)
    if cached is not None:
        return cached
    if canonical is None:
        return None
    plan = minimax_h3_resolve_plan(canonical)
    if isinstance(extra, dict):
        extra[MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY] = plan
    return plan


__all__ = [
    "MINIMAX_H3_BASE_SHORT_EDGE",
    "MINIMAX_H3_CANVAS_MULTIPLE",
    "MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY",
    "MINIMAX_H3_MAX_PIXELS",
    "MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY",
    "MiniMaxH3ResolvedPlan",
    "minimax_h3_plan_from_batch",
    "minimax_h3_resolve_plan",
    "minimax_h3_resolve_spatial_shape",
]
