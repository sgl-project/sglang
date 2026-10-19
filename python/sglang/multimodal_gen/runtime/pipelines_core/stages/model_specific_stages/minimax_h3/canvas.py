# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 keyframe target-canvas preparation.

Geometry behavior:
- auto-aspect canvases delegate to the shared adaptive v2 shape resolver;
- cover-crop: aspect-preserving max-scale LANCZOS resize + center crop,
  upscaling refused unless explicitly allowed.

Both the Qwen presentation (pixel_values) and the visual-condition tokenizer consume
the SAME prepared canvas image, so preparation
is cached per request in batch.extra.
"""

from __future__ import annotations

from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,
)

MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_PREPARED_KEYFRAMES_EXTRA_KEY = "minimax_h3_prepared_keyframes"


def minimax_h3_cover_crop_plan(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    allow_upscale: bool,
) -> dict[str, Any]:
    """Deterministic aspect-preserving cover-crop transform."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("cover_crop requires positive source width/height")
    scale = max(
        target_width / float(source_width), target_height / float(source_height)
    )
    if scale > 1.0 and not allow_upscale:
        raise ValueError(
            "target_canvas cover_crop would upscale the source; set "
            f"allow_upscale=true (source={source_width}x{source_height}, "
            f"target={target_width}x{target_height})"
        )
    resized_width = max(target_width, int(round(source_width * scale)))
    resized_height = max(target_height, int(round(source_height * scale)))
    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return {
        "scale": scale,
        "resized_size": (resized_width, resized_height),
        "crop_box": (left, top, left + target_width, top + target_height),
    }


def minimax_h3_prepare_keyframe_canvas(
    image: Any,
    *,
    target_width: int,
    target_height: int,
    allow_upscale: bool = False,
) -> Any:
    """Prepare a PIL image onto the target canvas.

    Identity (no resample) when the image already IS the canvas.
    """
    from PIL import Image

    image = image.convert("RGB")
    if image.size == (target_width, target_height):
        return image
    plan = minimax_h3_cover_crop_plan(
        source_width=image.size[0],
        source_height=image.size[1],
        target_width=target_width,
        target_height=target_height,
        allow_upscale=allow_upscale,
    )
    resized = image.resize(plan["resized_size"], Image.Resampling.LANCZOS)
    return resized.crop(plan["crop_box"])


def minimax_h3_stretch_keyframe_canvas(
    image: Any,
    *,
    target_width: int,
    target_height: int,
) -> Any:
    """Stretch the FL first frame directly onto the resolved target canvas."""

    from PIL import Image

    image = image.convert("RGB")
    if image.size == (target_width, target_height):
        return image
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _keyframe_materials(plan: Any) -> list[Any]:
    return [m for m in plan.materials if m.material_chain == "image.target_canvas"]


def _keyframe_canvas_size(shape: Any) -> tuple[int, int]:
    geometry = str(shape["geometry"])
    if geometry != "resolved_v2":
        raise ValueError(
            "keyframe preparation requires pre-queue resolved_v2 "
            f"geometry, got {geometry!r}"
        )
    return int(shape["width"]), int(shape["height"])


def _validate_keyframe_materials(plan: Any, keyframes: list[Any]) -> tuple[int, ...]:
    if str(plan.task) not in {"fl2va", "ref2va"}:
        raise ValueError(
            "keyframe target-canvas materials require plan.task='fl2va' or 'ref2va'"
        )
    semantic_indices = tuple(material.frame_index for material in keyframes)
    if semantic_indices not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            "MiniMax H3 keyframes must use one of the ordered frame_index signatures "
            f"{MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES!r}, got {semantic_indices!r}"
        )
    frame_count = plan.shape.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise ValueError("keyframe preparation requires an integer frame_count")
    if frame_count <= 1:
        raise ValueError("keyframe preparation requires frame_count > 1")
    expected_pixels = tuple(
        frame_count - 1 if index == -1 else index for index in semantic_indices
    )
    resolved_pixels = tuple(material.resolved_frame_index for material in keyframes)
    if resolved_pixels != expected_pixels:
        raise ValueError(
            "keyframe resolved_frame_index values disagree with semantic "
            f"anchors: expected {expected_pixels!r}, got {resolved_pixels!r}"
        )
    return semantic_indices


def minimax_h3_prepared_keyframes(batch: Any, plan: Any) -> dict[str, Any]:
    """Resolve + prepare one or two first/last keyframes once per request.

    The target canvas is shared across keyframes and must already be frozen by
    the pre-queue probe/resolve hook.
    Top-level ``image`` / ``canvas_width`` / ``canvas_height`` keys mirror the
    first-keyframe payload for compatibility; per-keyframe entries live under
    ``images``.
    """
    keyframes = _keyframe_materials(plan)
    semantic_indices = _validate_keyframe_materials(plan, keyframes)
    cached = batch.extra.get(MINIMAX_H3_PREPARED_KEYFRAMES_EXTRA_KEY)
    if cached is not None:
        cached_indices = tuple(cached.get("semantic_frame_indices") or ())
        cached_images = cached.get("images") or ()
        if cached_indices != semantic_indices or len(cached_images) != len(keyframes):
            raise ValueError(
                "cached keyframe preparation disagrees with the resolved plan"
            )
        return cached

    canvas_w, canvas_h = _keyframe_canvas_size(plan.shape)

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
        MINIMAX_H3_PROBE_FACTS_EXTRA_KEY,
        MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY,
    )

    probe_facts = batch.extra.get(MINIMAX_H3_PROBE_FACTS_EXTRA_KEY)
    material_shapes = batch.extra.get(MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY)
    for material in keyframes:
        condition_index = int(material.condition_index)
        facts = (
            probe_facts.get(condition_index) if isinstance(probe_facts, dict) else None
        )
        material_shape = (
            material_shapes.get(condition_index)
            if isinstance(material_shapes, dict)
            else None
        )
        if not isinstance(facts, dict) or not isinstance(material_shape, dict):
            raise ValueError(
                "fl2va keyframe preparation requires cached pre-queue probe and "
                f"shape facts for conditions[{condition_index}]"
            )
        if (
            int(material_shape.get("width") or 0),
            int(material_shape.get("height") or 0),
        ) != (canvas_w, canvas_h):
            raise ValueError(
                "fl2va keyframe material shape disagrees with the resolved target: "
                f"condition={condition_index}, material="
                f"{material_shape.get('width')}x{material_shape.get('height')}, "
                f"target={canvas_w}x{canvas_h}"
            )

    from PIL import Image, ImageOps

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
        minimax_h3_localize_material_uri,
    )

    entries: list[dict[str, Any]] = []
    for keyframe_index, material in enumerate(keyframes):
        image_path = minimax_h3_localize_material_uri(
            batch,
            material.uri,
            condition_type=material.condition_type,
            condition_index=int(material.condition_index),
        )
        with Image.open(image_path) as source_image:
            image = ImageOps.exif_transpose(source_image)
        # A request's first semantic keyframe is its geometry anchor, including
        # the single-image last-frame-only signature [-1]. Only the second image in the
        # two-keyframe FL extension is a follower and receives cover-crop.
        prepared_image = (
            minimax_h3_stretch_keyframe_canvas(
                image,
                target_width=canvas_w,
                target_height=canvas_h,
            )
            if keyframe_index == 0
            else minimax_h3_prepare_keyframe_canvas(
                image,
                target_width=canvas_w,
                target_height=canvas_h,
                allow_upscale=True,
            )
        )
        entries.append(
            {
                "image": prepared_image,
                "canvas_width": canvas_w,
                "canvas_height": canvas_h,
                "condition_index": int(material.condition_index),
                "frame_index": (
                    None if material.frame_index is None else int(material.frame_index)
                ),
                "resolved_frame_index": (
                    None
                    if material.resolved_frame_index is None
                    else int(material.resolved_frame_index)
                ),
            }
        )

    payload = {
        "image": entries[0]["image"],
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "images": entries,
        "semantic_frame_indices": [
            int(item["frame_index"])
            for item in entries
            if item.get("frame_index") is not None
        ],
        "pixel_frame_indices": [
            int(item["resolved_frame_index"])
            for item in entries
            if item.get("resolved_frame_index") is not None
        ],
        "frame_count": (
            int(plan.shape["frame_count"])
            if plan.shape.get("frame_count") is not None
            else None
        ),
    }
    batch.extra[MINIMAX_H3_PREPARED_KEYFRAMES_EXTRA_KEY] = payload
    return payload


__all__ = [
    "MINIMAX_H3_CANVAS_MULTIPLE",
    "MINIMAX_H3_PREPARED_KEYFRAMES_EXTRA_KEY",
    "minimax_h3_cover_crop_plan",
    "minimax_h3_prepare_keyframe_canvas",
    "minimax_h3_prepared_keyframes",
    "minimax_h3_stretch_keyframe_canvas",
]
