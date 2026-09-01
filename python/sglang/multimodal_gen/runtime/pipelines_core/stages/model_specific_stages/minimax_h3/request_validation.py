# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 canonical request validation.

Entry fail-fast for `minimax_h3.request/v1`: every violation raises ValueError
with the offending field path. Output is a normalized canonical dict (frame
indices validated but semantic -1 preserved, nothing else rewritten — prompt passes through verbatim and
conditions order is semantic, never reordered).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_MAX_DURATION_SECONDS,
    MINIMAX_H3_MIN_DURATION_SECONDS,
    MINIMAX_H3_RECOMMENDED_SHORT_EDGE,
    MINIMAX_H3_SUPPORTED_FPS,
    warn_unverified_short_edge,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_CONDITION_ROLE_KEYFRAME,
    MINIMAX_H3_CONDITION_ROLE_REFERENCE,
    MINIMAX_H3_FINITE_ASPECT_RATIOS,
    MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,
    MINIMAX_H3_TASK_FL2VA,
    MINIMAX_H3_TASK_REF2VA,
    MINIMAX_H3_TASK_T2VA,
    MiniMaxH3TaskProfile,
    canonical_minimax_h3_task,
    minimax_h3_task_profile,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_align_frame_count,
)

MINIMAX_H3_REQUEST_SCHEMA = "minimax_h3.request/v1"
MINIMAX_H3_MAX_SIGNED_SEED = (1 << 63) - 1
_ALLOWED_CONDITION_KEYS = frozenset(
    {"type", "uri", "role", "frame_index", "start_time_seconds"}
)


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _require_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _optional_positive_finite_float(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{path} must be a positive finite number")
    return normalized


def _optional_nonnegative_finite_float(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{path} must be a non-negative finite number")
    return normalized


def _validate_target(target: Any, *, profile: MiniMaxH3TaskProfile) -> dict[str, Any]:
    path = "target"
    if not isinstance(target, Mapping):
        raise ValueError(f"{path} is required and must be an object")
    # The canonical target has a deliberately small projection.  Transport
    # compatibility keys are ignored; only these three declared values are
    # validated and emitted below.
    short_edge = _require_int(target.get("short_edge"), f"{path}.short_edge")
    if short_edge <= 0:
        raise ValueError(f"{path}.short_edge must be positive, got {short_edge}")
    if short_edge != MINIMAX_H3_RECOMMENDED_SHORT_EDGE:
        # Same guard the resolver applies. Without it the recommended value warns
        # that it is "outside the verified configuration", naming itself as the
        # verified one.
        warn_unverified_short_edge(short_edge)
    aspect_ratio = _require_str(target.get("aspect_ratio"), f"{path}.aspect_ratio")
    if profile.aspect_ratio_forced_auto and aspect_ratio != "auto":
        raise ValueError(
            f'{path}.aspect_ratio must be "auto" for task {profile.task!r}, '
            f"got {aspect_ratio!r}"
        )
    has_duration = target.get("duration_seconds") is not None
    if (
        profile.task in {MINIMAX_H3_TASK_T2VA, MINIMAX_H3_TASK_REF2VA}
        and aspect_ratio != "auto"
        and aspect_ratio not in MINIMAX_H3_FINITE_ASPECT_RATIOS
    ):
        raise ValueError(
            f"{path}.aspect_ratio for task {profile.task!r} must be 'auto' or "
            f"one of {list(MINIMAX_H3_FINITE_ASPECT_RATIOS)!r}, got "
            f"{aspect_ratio!r}"
        )
    if not has_duration:
        if not profile.duration_from_audio_reference:
            raise ValueError(f"{path}.duration_seconds is required")
        # ref2va: duration may derive from a reference audio; the
        # audio-condition presence is enforced after conditions validate.
    out: dict[str, Any] = {
        "short_edge": short_edge,
        "aspect_ratio": aspect_ratio,
    }
    if has_duration:
        duration = target["duration_seconds"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError(f"{path}.duration_seconds must be a number")
        if duration <= 0:
            raise ValueError(f"{path}.duration_seconds must be positive")
        if not (
            MINIMAX_H3_MIN_DURATION_SECONDS
            <= float(duration)
            <= MINIMAX_H3_MAX_DURATION_SECONDS
        ):
            raise ValueError(
                f"{path}.duration_seconds must be in "
                f"[{MINIMAX_H3_MIN_DURATION_SECONDS:g}, "
                f"{MINIMAX_H3_MAX_DURATION_SECONDS:g}], got {duration}"
            )
        out["duration_seconds"] = float(duration)
    return out


def _validate_conditions(
    conditions: Any,
    *,
    profile: MiniMaxH3TaskProfile,
    frame_count: int | None,
) -> list[dict[str, Any]]:
    path = "conditions"
    if conditions is None:
        conditions = []
    if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
        raise ValueError(f"{path} must be a list")

    if not profile.conditions_required:
        if len(conditions) > 0:
            raise ValueError(
                f"{path} must be empty for task {profile.task!r} "
                f"(got {len(conditions)} entries)"
            )
        return []
    if len(conditions) == 0:
        raise ValueError(
            f"{path} requires at least one entry for task {profile.task!r}"
        )
    if (
        profile.min_condition_count is not None
        and len(conditions) < profile.min_condition_count
    ):
        raise ValueError(
            f"{path} requires at least {profile.min_condition_count} entries "
            f"for task {profile.task!r}, got {len(conditions)}"
        )
    if (
        profile.max_condition_count is not None
        and len(conditions) > profile.max_condition_count
    ):
        raise ValueError(
            f"{path} allows at most {profile.max_condition_count} entries "
            f"for task {profile.task!r}, got {len(conditions)}"
        )

    aligned_frame_count = (
        minimax_h3_align_frame_count(frame_count) if frame_count is not None else None
    )
    normalized: list[dict[str, Any]] = []
    seen_frame_indices: dict[int, int] = {}
    for index, cond in enumerate(conditions):
        cpath = f"{path}[{index}]"
        if not isinstance(cond, Mapping):
            raise ValueError(f"{cpath} must be an object")
        unknown = set(cond) - _ALLOWED_CONDITION_KEYS
        if unknown:
            raise ValueError(f"{cpath} has unknown fields: {sorted(unknown)}")
        role = _require_str(cond.get("role"), f"{cpath}.role")
        if role not in (
            MINIMAX_H3_CONDITION_ROLE_KEYFRAME,
            MINIMAX_H3_CONDITION_ROLE_REFERENCE,
        ):
            raise ValueError(
                f"{cpath}.role must be keyframe or reference, " f"got {role!r}"
            )
        cond_type = _require_str(cond.get("type"), f"{cpath}.type")
        try:
            rule = profile.rule_for(role=role, condition_type=cond_type)
        except ValueError as exc:
            raise ValueError(f"{cpath}: {exc}") from exc
        uri = _require_str(cond.get("uri"), f"{cpath}.uri")

        entry: dict[str, Any] = {"type": cond_type, "uri": uri, "role": role}
        if rule.requires_frame_index:
            frame_index = _require_int(cond.get("frame_index"), f"{cpath}.frame_index")
            if aligned_frame_count is None:
                raise ValueError(
                    f"{cpath}.frame_index requires a resolved target duration"
                )
            if frame_index == -1:
                resolved = aligned_frame_count - 1
            elif 0 <= frame_index < aligned_frame_count:
                resolved = frame_index
            else:
                raise ValueError(
                    f"{cpath}.frame_index must be -1 or in "
                    f"[0, {aligned_frame_count}) after 17n+5 frame alignment, "
                    f"got {frame_index}"
                )
            if resolved in seen_frame_indices:
                raise ValueError(
                    f"{cpath}.frame_index resolves to {resolved}, already "
                    f"bound by conditions[{seen_frame_indices[resolved]}]"
                )
            seen_frame_indices[resolved] = index
            # Preserve the request-level semantic index.  In particular, -1 is
            # the canonical last-frame sentinel; the resolved pixel frame is
            # carried separately by MiniMaxH3ResolvedPlan.
            entry["frame_index"] = frame_index
        elif cond.get("frame_index") is not None:
            raise ValueError(f"{cpath}.frame_index is not allowed for role={role!r}")
        start_time_seconds = _optional_nonnegative_finite_float(
            cond.get("start_time_seconds"), f"{cpath}.start_time_seconds"
        )
        if start_time_seconds is not None:
            if cond_type not in {"video", "video_audio"}:
                raise ValueError(
                    f"{cpath}.start_time_seconds is only allowed for video "
                    "or video_audio references"
                )
            entry["start_time_seconds"] = start_time_seconds
        normalized.append(entry)
    return normalized


def _validate_keyframe_conditions(
    conditions: Sequence[Mapping[str, Any]], *, task: str
) -> None:
    """Enforce the shared first/last-frame contract after schema validation."""

    keyframes = [
        condition
        for condition in conditions
        if condition["role"] == MINIMAX_H3_CONDITION_ROLE_KEYFRAME
    ]
    frame_indices = tuple(condition.get("frame_index") for condition in keyframes)
    if frame_indices not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            f"conditions for task {task!r} must include one or two ordered "
            "image/keyframe entries with frame_index [0], [-1], or [0, -1], "
            f"got {list(frame_indices)!r}"
        )
    if task == MINIMAX_H3_TASK_REF2VA and not any(
        condition["role"] == MINIMAX_H3_CONDITION_ROLE_REFERENCE
        for condition in conditions
    ):
        raise ValueError(
            "ref2va keyframes require at least one reference condition; "
            "use task 'fl2va' for keyframe-only generation"
        )


def minimax_h3_validate_canonical_request(
    *,
    task: Any,
    prompt: Any,
    conditions: Any,
    target: Any,
    flow_shift: Any = None,
    audio_flow_shift: Any = None,
    seed: Any = None,
    **_extra_kwargs: Any,
) -> dict[str, Any]:
    """Validate and normalize a `minimax_h3.request/v1` canonical request.

    Returns the normalized canonical dict; raises ValueError with a field
    path on any violation. Conditions order is preserved (it is semantic:
    prompt ordinal labels reference it). seed=0 is a legal value.
    """
    # Accept transport wrappers and compatibility kwargs at this boundary, but
    # never copy them into the canonical request.
    del _extra_kwargs
    # Normalize the task name before profile lookup so offline callers match
    # the adapter behaviour.
    task_name = canonical_minimax_h3_task(_require_str(task, "task"))
    profile = minimax_h3_task_profile(task_name)
    prompt_text = _require_str(prompt, "prompt")

    normalized_target = _validate_target(target, profile=profile)
    requested_frame_count = None
    if normalized_target.get("duration_seconds") is not None:
        requested_frame_count = int(
            round(
                float(normalized_target["duration_seconds"]) * MINIMAX_H3_SUPPORTED_FPS
            )
        )
    normalized_conditions = _validate_conditions(
        conditions,
        profile=profile,
        frame_count=requested_frame_count,
    )
    if profile.task == MINIMAX_H3_TASK_FL2VA:
        _validate_keyframe_conditions(normalized_conditions, task=profile.task)
    elif profile.task == MINIMAX_H3_TASK_REF2VA and any(
        condition["role"] == MINIMAX_H3_CONDITION_ROLE_KEYFRAME
        for condition in normalized_conditions
    ):
        _validate_keyframe_conditions(normalized_conditions, task=profile.task)
    # ref2va accepts ordered reference streams and, for hybrid checkpoints,
    # one first/last keyframe signature. Type admission is handled by the task
    # profile; temporal ambiguity is validated later when target duration is
    # omitted.
    if not profile.video_reference_supported:
        for index, cond in enumerate(normalized_conditions):
            if cond["type"] in ("video", "video_audio"):
                raise ValueError(
                    f"conditions[{index}]: video references are not supported "
                    f"in v1 for task {profile.task!r} (image/audio only)"
                )
    if normalized_target.get("duration_seconds") is None:
        # Only reachable for duration_from_audio_reference profiles.
        duration_sources = [
            cond
            for cond in normalized_conditions
            if cond["type"] in ("audio", "video", "video_audio")
        ]
        if not duration_sources:
            raise ValueError(
                "target.duration_seconds is required, or exactly one "
                "audio reference to derive duration from (including "
                f"video/video_audio soundtracks; task {profile.task!r})"
            )
        if len(duration_sources) > 1:
            raise ValueError(
                "target.duration_seconds is required when multiple "
                "audio-bearing references are provided"
            )

    canonical: dict[str, Any] = {
        "schema": MINIMAX_H3_REQUEST_SCHEMA,
        "task": task_name,
        "prompt": prompt_text,
        "conditions": normalized_conditions,
        "target": normalized_target,
    }
    normalized_flow_shift = _optional_positive_finite_float(flow_shift, "flow_shift")
    normalized_audio_flow_shift = _optional_positive_finite_float(
        audio_flow_shift, "audio_flow_shift"
    )
    if normalized_flow_shift is not None:
        canonical["flow_shift"] = normalized_flow_shift
    if normalized_audio_flow_shift is not None:
        canonical["audio_flow_shift"] = normalized_audio_flow_shift
    if seed is not None:
        normalized_seed = _require_int(seed, "seed")
        if normalized_seed < 0:
            raise ValueError(f"seed must be non-negative, got {normalized_seed}")
        if normalized_seed > MINIMAX_H3_MAX_SIGNED_SEED:
            raise ValueError(
                f"seed must not exceed the signed int64 maximum, got {normalized_seed}"
            )
        canonical["seed"] = normalized_seed
    return canonical


__all__ = [
    "MINIMAX_H3_REQUEST_SCHEMA",
    "MINIMAX_H3_MAX_SIGNED_SEED",
    "MINIMAX_H3_SUPPORTED_FPS",
    "minimax_h3_validate_canonical_request",
]
