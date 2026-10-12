# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 probe -> resolve-once admission hook.

This module is intentionally data/CPU only. It localizes condition media,
caches display-geometry facts, freezes every target/material canvas, and
resolves the real aligned workload before a video job is published or sent to
the scheduler.
"""

from __future__ import annotations

from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_MAX_DURATION_SECONDS,
    MINIMAX_H3_MIN_DURATION_SECONDS,
    MINIMAX_H3_SUPPORTED_FPS,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
    MINIMAX_H3_TEMP_DIRS_EXTRA_KEY,
    minimax_h3_cleanup_temp_dirs,
    minimax_h3_probe_material,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_BASE_SHORT_EDGE,
    MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY,
    MiniMaxH3ResolvedPlan,
    minimax_h3_plan_from_batch,
    minimax_h3_resolve_spatial_shape,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_align_frame_count,
    minimax_h3_audio_latent_t,
    minimax_h3_video_latent_t,
)

MINIMAX_H3_PROBE_FACTS_EXTRA_KEY = "minimax_h3_probe_facts_by_condition"
MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY = "minimax_h3_resolved_material_shapes"


def _replace_plan_shape(
    plan: MiniMaxH3ResolvedPlan, shape: dict[str, Any]
) -> MiniMaxH3ResolvedPlan:
    return MiniMaxH3ResolvedPlan(
        task=plan.task,
        prompt=plan.prompt,
        seed=plan.seed,
        materials=plan.materials,
        encoders=plan.encoders,
        branches=plan.branches,
        default_flow_shift=plan.default_flow_shift,
        default_audio_flow_shift=plan.default_audio_flow_shift,
        flow_shift=plan.flow_shift,
        audio_flow_shift=plan.audio_flow_shift,
        shape=shape,
        condition_mask=plan.condition_mask,
    )


def _display_shape(facts: dict[str, Any], *, label: str) -> tuple[float, float]:
    try:
        width = float(facts["display_width"])
        height = float(facts["display_height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} has no usable display geometry") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} has no usable display geometry")
    return width, height


def _resolve_deferred_spatial_shape(
    plan: MiniMaxH3ResolvedPlan,
    shape: dict[str, Any],
    probe_facts: dict[int, dict[str, Any]],
) -> None:
    if str(shape.get("geometry")) != "deferred":
        return
    if plan.task == "fl2va":
        candidates = [
            material
            for material in plan.materials
            if material.material_chain == "image.target_canvas"
        ]
        if len(candidates) not in {1, 2}:
            raise ValueError(
                f"fl2va requires one or two keyframe materials, got {len(candidates)}"
            )
        for material in candidates:
            if material.frame_index not in {0, -1}:
                raise ValueError(
                    "fl2va deferred geometry requires semantic frame_index 0 or "
                    f"-1, got {material.frame_index!r} for "
                    f"conditions[{material.condition_index}]"
                )
        # Select by semantic time, not request/material iteration order. The
        # last-frame sentinel sorts after the first-frame anchor.
        source = min(
            candidates,
            key=lambda material: (
                material.frame_index == -1,
                int(material.frame_index),
                int(material.condition_index),
            ),
        )
        geometry_source = (
            "first_keyframe" if source.frame_index == 0 else "last_keyframe"
        )
    else:
        raise ValueError(f"task {plan.task!r} has unsupported deferred target geometry")
    width, height = _display_shape(
        probe_facts[int(source.condition_index)],
        label=f"conditions[{source.condition_index}]",
    )
    shape.update(
        minimax_h3_resolve_spatial_shape(
            width=width,
            height=height,
            base_short_edge=int(shape["base_short_edge"]),
        )
    )
    shape["geometry_source"] = geometry_source
    shape["geometry_source_condition_index"] = int(source.condition_index)
    if source.frame_index is not None:
        shape["geometry_source_frame_index"] = int(source.frame_index)


def _resolve_deferred_temporal_shape(
    plan: MiniMaxH3ResolvedPlan,
    shape: dict[str, Any],
    probe_facts: dict[int, dict[str, Any]],
) -> None:
    if str(shape.get("temporal")) != "deferred_from_audio_reference":
        return
    sources = [
        material
        for material in plan.materials
        if material.condition_type in {"audio", "video", "video_audio"}
        and bool(probe_facts[int(material.condition_index)].get("has_audio"))
    ]
    if len(sources) != 1:
        raise ValueError(
            "audio-derived target duration requires exactly one probed "
            f"condition with an audio stream, got {len(sources)}"
        )
    source = sources[0]
    facts = probe_facts[int(source.condition_index)]
    try:
        duration_seconds = float(facts["audio_duration_seconds"]) - float(
            source.start_time_seconds
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("audio reference has no positive probed duration") from exc
    if duration_seconds <= 0:
        raise ValueError("audio reference has no positive probed duration")
    if not (
        MINIMAX_H3_MIN_DURATION_SECONDS
        <= duration_seconds
        <= MINIMAX_H3_MAX_DURATION_SECONDS
    ):
        raise ValueError(
            "audio reference duration must be in "
            f"[{MINIMAX_H3_MIN_DURATION_SECONDS:g}, "
            f"{MINIMAX_H3_MAX_DURATION_SECONDS:g}] seconds, got {duration_seconds:g}"
        )
    fps = MINIMAX_H3_SUPPORTED_FPS
    frame_count = minimax_h3_align_frame_count(int(round(duration_seconds * fps)))
    aligned_duration = frame_count / fps
    shape.update(
        {
            "temporal": "resolved_from_audio_reference",
            "duration_seconds": aligned_duration,
            "frame_count": frame_count,
            "video_latent_t": minimax_h3_video_latent_t(frame_count),
            "audio_latent_t": minimax_h3_audio_latent_t(aligned_duration),
        }
    )


def _validate_reference_start_times(
    plan: MiniMaxH3ResolvedPlan,
    probe_facts: dict[int, dict[str, Any]],
) -> None:
    for material in plan.materials:
        start_time_seconds = float(material.start_time_seconds)
        if start_time_seconds == 0:
            continue
        condition_index = int(material.condition_index)
        facts = probe_facts[condition_index]
        try:
            video_duration_seconds = float(facts["video_duration_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"conditions[{condition_index}].start_time_seconds requires "
                "a video with a positive probed duration"
            ) from exc
        if start_time_seconds >= video_duration_seconds:
            raise ValueError(
                f"conditions[{condition_index}].start_time_seconds must be less "
                f"than the video duration {video_duration_seconds:g}, got "
                f"{start_time_seconds:g}"
            )
        if bool(facts.get("has_audio")):
            try:
                audio_duration_seconds = float(facts["audio_duration_seconds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"conditions[{condition_index}] has no usable audio duration"
                ) from exc
            if start_time_seconds >= audio_duration_seconds:
                raise ValueError(
                    f"conditions[{condition_index}].start_time_seconds must be "
                    f"less than the soundtrack duration {audio_duration_seconds:g}, "
                    f"got {start_time_seconds:g}"
                )


def _resolved_work_frame_count(
    plan: MiniMaxH3ResolvedPlan,
    shape: dict[str, Any],
    probe_facts: dict[int, dict[str, Any]],
) -> int:
    if shape.get("frame_count") is not None:
        return int(shape["frame_count"])
    raise ValueError("MiniMax H3 target frame count remained unresolved before queue")


def _preserve_prequeue_material_dirs(batch: Any) -> None:
    temp_registry = batch.extra.get(MINIMAX_H3_TEMP_DIRS_EXTRA_KEY)
    if isinstance(temp_registry, dict) and "material" in temp_registry:
        # Multi-output dispatch shallow-copies request extras. Keep the
        # single localized source closure owned by the API request until
        # every output finishes; encoder-stage "material" cleanup must not
        # delete it after the first expanded output.
        paths = temp_registry.pop("material")
        prequeue_paths = temp_registry.setdefault("prequeue_material", [])
        for path in paths if isinstance(paths, list) else ():
            if path not in prequeue_paths:
                prequeue_paths.append(path)


def minimax_h3_prepare_for_queue(batch: Any) -> MiniMaxH3ResolvedPlan:
    """Freeze MiniMax H3 media/shape facts before queue admission."""

    try:
        plan = minimax_h3_plan_from_batch(batch)
        if plan is None:
            raise ValueError(
                "MiniMax H3 pre-queue validation requires a canonical request"
            )

        probe_facts: dict[int, dict[str, Any]] = {}
        for material in plan.materials:
            probe_facts[int(material.condition_index)] = minimax_h3_probe_material(
                batch,
                material.uri,
                condition_type=material.condition_type,
                condition_index=int(material.condition_index),
            )
        batch.extra[MINIMAX_H3_PROBE_FACTS_EXTRA_KEY] = probe_facts

        shape = dict(plan.shape)
        _validate_reference_start_times(plan, probe_facts)
        _resolve_deferred_spatial_shape(plan, shape, probe_facts)
        _resolve_deferred_temporal_shape(plan, shape, probe_facts)
        if str(shape.get("geometry")) != "resolved_v2":
            raise ValueError(
                "MiniMax H3 target geometry remained unresolved before queue"
            )

        material_shapes: dict[int, dict[str, Any]] = {}
        for material in plan.materials:
            condition_index = int(material.condition_index)
            if material.material_chain == "image.target_canvas":
                resolved = {
                    key: shape[key]
                    for key in (
                        "geometry",
                        "shape_policy_version",
                        "base_short_edge",
                        "effective_short_edge",
                        "size_mode",
                        "max_pixels",
                        "multiple",
                        "rounding",
                        "width",
                        "height",
                    )
                    if key in shape
                }
            elif material.material_chain in {
                "video.reference_preserve",
                "video_audio.reference_preserve",
            }:
                width, height = _display_shape(
                    probe_facts[condition_index],
                    label=f"conditions[{condition_index}]",
                )
                resolved = minimax_h3_resolve_spatial_shape(
                    width=width,
                    height=height,
                    base_short_edge=MINIMAX_H3_BASE_SHORT_EDGE,
                )
            elif material.material_chain == "image.reference_preserve":
                width, height = _display_shape(
                    probe_facts[condition_index],
                    label=f"conditions[{condition_index}]",
                )
                from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
                    minimax_h3_resolve_reference_image_shape,
                )

                resolved = minimax_h3_resolve_reference_image_shape(
                    width=width,
                    height=height,
                )
            else:
                continue
            resolved = dict(resolved)
            resolved["condition_index"] = condition_index
            material_shapes[condition_index] = resolved
        batch.extra[MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY] = material_shapes

        work_frames = _resolved_work_frame_count(plan, shape, probe_facts)
        resolved_plan = _replace_plan_shape(plan, shape)
        batch.extra[MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY] = resolved_plan
        # Req delegates these fields to SamplingParams. Freezing them here makes
        # queue metadata and dynamic-batch signatures use the same final shape
        # that the MiniMax H3 stages consume.
        batch.width = int(shape["width"])
        batch.height = int(shape["height"])
        batch.fps = MINIMAX_H3_SUPPORTED_FPS
        batch.num_frames = int(work_frames)
        _preserve_prequeue_material_dirs(batch)
        return resolved_plan
    except Exception:
        minimax_h3_cleanup_temp_dirs(batch)
        batch.extra.pop(MINIMAX_H3_PROBE_FACTS_EXTRA_KEY, None)
        batch.extra.pop(MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY, None)
        raise


__all__ = [
    "MINIMAX_H3_PROBE_FACTS_EXTRA_KEY",
    "MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY",
    "minimax_h3_prepare_for_queue",
]
