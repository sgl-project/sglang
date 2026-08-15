# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 request-local noise preparation.

MiniMax H3 consumes one CPU generator per request. Visual condition noise is
drawn first, one tensor per condition in packed order, followed by the target
video tensor and the target audio rows. Keeping that order in one place makes
the seed contract independent of pipeline stage boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
)

MINIMAX_H3_AUDIO_COND_CHANNELS = 2


def minimax_h3_resolve_condition_noise_aug(sampling: Any) -> tuple[float, float]:
    """Resolve visual/audio condition timesteps using the model defaults."""

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
        MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
        MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )

    imgvid_noise_aug = getattr(sampling, "imgvid_cond_noise_aug_for_inference", None)
    if imgvid_noise_aug is None:
        imgvid_noise_aug = MINIMAX_H3_IMGVID_COND_TIMESTEP
    audio_noise_aug = getattr(sampling, "audio_cond_noise_aug_for_inference", None)
    if audio_noise_aug is None:
        audio_noise_aug = MINIMAX_H3_AUDIO_REF_COND_TIMESTEP
    return _validate_noise_aug(imgvid_noise_aug), _validate_noise_aug(audio_noise_aug)


def minimax_h3_ref_payload_entry(
    payload: Any,
    *,
    list_key: str,
    condition_index: int,
    path: str,
) -> Mapping[str, Any]:
    """Select one encoded reference payload by canonical condition index."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} is required for ref2va condition rows")
    entries = payload.get(list_key)
    if isinstance(entries, list):
        for entry in entries:
            if (
                isinstance(entry, Mapping)
                and entry.get("condition_index") is not None
                and int(entry["condition_index"]) == int(condition_index)
            ):
                return entry
        if len(entries) == 1 and isinstance(entries[0], Mapping):
            return entries[0]
        raise ValueError(
            f"{path}.{list_key} missing entry for condition_index={condition_index}"
        )
    if payload.get("condition_index") is None or int(payload["condition_index"]) == int(
        condition_index
    ):
        return payload
    raise ValueError(
        f"{path}.{list_key} missing entry for condition_index={condition_index}"
    )


def minimax_h3_condition_noise_shapes(
    batch: Any, plan: Any
) -> tuple[list[tuple[int, int, int]], list[int]]:
    """Return condition shapes in the same order as the packed request."""

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY,
        MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY,
        MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY,
        MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY,
    )

    task = str(plan.task)
    if task == "t2va":
        return [], []
    if task == "fl2va":
        payload = batch.extra.get(MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY)
        if not isinstance(payload, Mapping):
            raise ValueError("fl2va noise preparation requires encoded keyframes")
        entries = payload.get("keyframes")
        if not isinstance(entries, list) or not entries:
            raise ValueError("fl2va keyframe payload must carry ordered keyframes")
        return [
            (1, int(entry["latent_h"]), int(entry["latent_w"])) for entry in entries
        ], []
    if task != "ref2va":
        raise ValueError(f"unsupported MiniMax H3 task {task!r}")

    ref_image = batch.extra.get(MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY)
    ref_audio = batch.extra.get(MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY)
    ref_video = batch.extra.get(MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY)
    visual_shapes: list[tuple[int, int, int]] = []
    audio_lengths: list[int] = []
    for material in plan.materials:
        chain = str(material.material_chain)
        condition_index = int(material.condition_index)
        if chain == "image.reference_preserve":
            entry = minimax_h3_ref_payload_entry(
                ref_image,
                list_key="images",
                condition_index=condition_index,
                path="batch.extra.minimax_h3_reference_image_rows",
            )
            visual_shapes.append((1, int(entry["latent_h"]), int(entry["latent_w"])))
        elif chain == "audio":
            entry = minimax_h3_ref_payload_entry(
                ref_audio,
                list_key="audios",
                condition_index=condition_index,
                path="batch.extra.minimax_h3_reference_audio_rows",
            )
            if int(entry["ref_audio_t"]) > 0:
                audio_lengths.append(int(entry["ref_audio_t"]))
        elif chain in {"video.reference_preserve", "video_audio.reference_preserve"}:
            video_entry = minimax_h3_ref_payload_entry(
                ref_video,
                list_key="videos",
                condition_index=condition_index,
                path="batch.extra.minimax_h3_reference_video_rows",
            )
            audio_entry = minimax_h3_ref_payload_entry(
                ref_audio,
                list_key="audios",
                condition_index=condition_index,
                path="batch.extra.minimax_h3_reference_audio_rows",
            )
            visual_shapes.append(
                (
                    int(video_entry["latent_t"]),
                    int(video_entry["latent_h"]),
                    int(video_entry["latent_w"]),
                )
            )
            if int(audio_entry["ref_audio_t"]) > 0:
                audio_lengths.append(int(audio_entry["ref_audio_t"]))
        else:
            raise ValueError(f"unsupported ref2va material chain {chain!r}")
    return visual_shapes, audio_lengths


def minimax_h3_prepare_request_noise(
    *,
    seed: int,
    condition_shapes: Sequence[Sequence[int]],
    condition_audio_t: Sequence[int],
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    imgvid_noise_aug: float,
    audio_noise_aug: float,
) -> dict[str, torch.Tensor | None]:
    """Draw all request noise from one CPU generator in reference order."""

    imgvid_noise_aug = _validate_noise_aug(imgvid_noise_aug)
    audio_noise_aug = _validate_noise_aug(audio_noise_aug)
    parsed_shapes = [_validate_visual_shape(shape) for shape in condition_shapes]
    parsed_audio_t = [int(value) for value in condition_audio_t]
    if any(value <= 0 for value in parsed_audio_t):
        raise ValueError(
            f"condition audio latent lengths must be positive, got {parsed_audio_t}"
        )
    latent_t, latent_h, latent_w = _validate_visual_shape(
        (latent_t, latent_h, latent_w)
    )
    audio_t = int(audio_t)
    if audio_t <= 0:
        raise ValueError(f"audio_t must be positive, got {audio_t}")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    condition_video_noise_rows = None
    if parsed_shapes and imgvid_noise_aug < 1.0:
        rows = [
            minimax_h3_patchify_video_latent(
                torch.randn(
                    1,
                    24,
                    shape_t,
                    shape_h,
                    shape_w,
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                ),
                patch_size=[1, 2, 2],
            )
            for shape_t, shape_h, shape_w in parsed_shapes
        ]
        condition_video_noise_rows = rows[0] if len(rows) == 1 else torch.cat(rows)

    condition_audio_noise_rows = None
    if parsed_audio_t and audio_noise_aug < 1.0:
        rows = [
            torch.randn(
                MINIMAX_H3_AUDIO_COND_CHANNELS * condition_t,
                32,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
            for condition_t in parsed_audio_t
        ]
        condition_audio_noise_rows = rows[0] if len(rows) == 1 else torch.cat(rows)

    video_tensor = torch.randn(
        1,
        24,
        latent_t,
        latent_h,
        latent_w,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    initial_video_rows = minimax_h3_patchify_video_latent(
        video_tensor, patch_size=[1, 2, 2]
    ).to(torch.float32)
    initial_audio_rows = torch.randn(
        MINIMAX_H3_AUDIO_COND_CHANNELS * audio_t,
        32,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return {
        "condition_video_noise_rows": condition_video_noise_rows,
        "condition_audio_noise_rows": condition_audio_noise_rows,
        "initial_video_rows": initial_video_rows,
        "initial_audio_rows": initial_audio_rows,
    }


def minimax_h3_imgvid_cond_noise_aug_rows(
    clean_rows: torch.Tensor,
    *,
    noise_rows: torch.Tensor,
    noise_aug: float,
) -> torch.Tensor:
    return _mix_condition_rows(
        clean_rows,
        noise_rows=noise_rows,
        noise_aug=noise_aug,
        width=96,
        name="imgvid",
    )


def minimax_h3_audio_cond_noise_aug_rows(
    clean_rows: torch.Tensor,
    *,
    noise_rows: torch.Tensor,
    noise_aug: float,
) -> torch.Tensor:
    return _mix_condition_rows(
        clean_rows,
        noise_rows=noise_rows,
        noise_aug=noise_aug,
        width=32,
        name="audio",
    )


def _mix_condition_rows(
    clean_rows: torch.Tensor,
    *,
    noise_rows: torch.Tensor,
    noise_aug: float,
    width: int,
    name: str,
) -> torch.Tensor:
    noise_aug = _validate_noise_aug(noise_aug)
    if clean_rows.ndim != 2 or int(clean_rows.shape[1]) != width:
        raise ValueError(
            f"clean {name} condition rows must have shape [n, {width}], "
            f"got {list(clean_rows.shape)}"
        )
    if noise_rows.shape != clean_rows.shape:
        raise ValueError(
            f"{name} condition noise shape {list(noise_rows.shape)} != "
            f"clean rows {list(clean_rows.shape)}"
        )
    clean = clean_rows.to(dtype=torch.float32)
    noise = noise_rows.to(device=clean.device, dtype=torch.float32)
    timestep = torch.tensor(noise_aug, dtype=torch.float32, device=clean.device)
    return (timestep * clean + (1.0 - timestep) * noise).contiguous()


def _validate_visual_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(
            "visual latent shape must be (latent_t, latent_h, latent_w), "
            f"got {list(shape)}"
        )
    latent_t, latent_h, latent_w = (int(value) for value in shape)
    if latent_t <= 0 or latent_h <= 0 or latent_w <= 0:
        raise ValueError(f"visual latent shape must be positive, got {list(shape)}")
    if latent_h % 2 or latent_w % 2:
        raise ValueError(
            "visual latent spatial dimensions must be divisible by 2, "
            f"got {(latent_t, latent_h, latent_w)}"
        )
    return latent_t, latent_h, latent_w


def _validate_noise_aug(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"noise_aug must be in [0, 1], got {value}")
    return value


__all__ = [
    "minimax_h3_audio_cond_noise_aug_rows",
    "minimax_h3_condition_noise_shapes",
    "minimax_h3_imgvid_cond_noise_aug_rows",
    "minimax_h3_prepare_request_noise",
    "minimax_h3_ref_payload_entry",
    "minimax_h3_resolve_condition_noise_aug",
]
