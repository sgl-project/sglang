# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Runtime utility surface for SANA-WM stages."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

TARGET_HEIGHT = 704
TARGET_WIDTH = 1280
# Official NVlabs defaults (utils.py upstream). FOOTGUN: 0.05 is the
# BIDIRECTIONAL default; the streaming pipelines use 0.04
# (STREAMING_TRANSLATION_SPEED == _SANA_WM_DEFAULT_TRANSLATION_SPEED in
# base.py). Every runtime call site passes translation_speed
# explicitly — keep it that way; do not rely on this module default.
DEFAULT_TRANSLATION_SPEED = 0.05
DEFAULT_ROTATION_SPEED_DEG = 1.2
DEFAULT_PITCH_LIMIT_DEG = 85.0
ALLOWED_ACTION_KEYS = frozenset("wasdijkl")


def pil_to_model_tensor(
    image: Image.Image, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor * 2.0 - 1.0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def compute_resize_crop_geometry(
    src_w: int, src_h: int, target_h: int, target_w: int
) -> tuple[int, int, int, int]:
    """Aspect-preserving resize + center-crop geometry; returns ``(resized_w, resized_h, left, top)``.

    Single source for the three resize/crop implementations (utils PIL, batch PIL, batch tensor).
    """
    scale = max(target_h / float(src_h), target_w / float(src_w))
    resized_w = max(target_w, int(round(src_w * scale)))
    resized_h = max(target_h, int(round(src_h * scale)))
    left = (resized_w - target_w) // 2
    top = (resized_h - target_h) // 2
    return resized_w, resized_h, left, top


def normalize_camera_actions(
    payload: Any,
    *,
    allow_none: bool = False,
    error_label: str = "camera_actions",
) -> list[list[str]]:
    """Validate + lowercase-normalize a ``list[list[str]]`` camera-action payload.

    Shared by the realtime adapter (``allow_none=False``) and realtime stage (``allow_none=True``).
    """
    if payload is None and allow_none:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"{error_label} must be list[list[str]]")
    out: list[list[str]] = []
    for frame_actions in payload:
        if not isinstance(frame_actions, list):
            raise ValueError(f"{error_label} must be list[list[str]]")
        out.append([str(key).lower() for key in frame_actions])
    return out


def parse_action_string(action: str) -> list[list[str]]:
    cleaned = "".join(action.replace("，", ",").split())
    if not cleaned:
        raise ValueError("action string is empty")

    per_frame: list[list[str]] = []
    for segment in cleaned.split(","):
        if not segment or "-" not in segment:
            raise ValueError(
                f"invalid action segment {segment!r}; expected '<keys>-<frames>'"
            )
        keys_part, duration = segment.rsplit("-", 1)
        if not duration.isdigit() or int(duration) <= 0:
            raise ValueError(f"invalid duration in action segment {segment!r}")

        if keys_part.lower() == "none":
            keys: list[str] = []
        else:
            bad = sorted(
                {char for char in keys_part.lower() if char not in ALLOWED_ACTION_KEYS}
            )
            if bad:
                raise ValueError(
                    f"unknown action keys {bad}; allowed keys are {sorted(ALLOWED_ACTION_KEYS)}"
                )
            keys = sorted(set(keys_part.lower()))
        # Fresh list per frame: repeated frames must NOT alias one list object
        # (callers could mutate a frame and silently edit all its repeats).
        per_frame.extend([list(keys) for _ in range(int(duration))])
    return per_frame


def action_string_to_c2w(
    action: str,
    *,
    translation_speed: float = DEFAULT_TRANSLATION_SPEED,
    rotation_speed_deg: float = DEFAULT_ROTATION_SPEED_DEG,
    pitch_limit_deg: float = DEFAULT_PITCH_LIMIT_DEG,
    strafe_yaw_coupling: float = 0.4,
) -> np.ndarray:
    per_frame = parse_action_string(action)
    rotate_rad = math.radians(rotation_speed_deg)
    pitch_limit_rad = math.radians(pitch_limit_deg)
    current = np.eye(4, dtype=np.float64)
    poses = [current.copy()]
    current_pitch = 0.0

    for keys in per_frame:
        held = set(keys)
        rotation = current[:3, :3]
        translation = current[:3, 3]

        pitch_delta = (rotate_rad if "i" in held else 0.0) - (
            rotate_rad if "k" in held else 0.0
        )
        next_pitch = current_pitch + pitch_delta
        if -pitch_limit_rad <= next_pitch <= pitch_limit_rad:
            current_pitch = next_pitch
        else:
            pitch_delta = 0.0

        yaw_delta = (rotate_rad if "l" in held else 0.0) - (
            rotate_rad if "j" in held else 0.0
        )
        strafe_yaw = (rotate_rad if "d" in held else 0.0) - (
            rotate_rad if "a" in held else 0.0
        )
        yaw_delta += strafe_yaw_coupling * strafe_yaw
        rotation = _rot_y(yaw_delta) @ rotation @ _rot_x(pitch_delta)

        forward = rotation[:, 2].copy()
        right = rotation[:, 0].copy()
        forward[1] = 0.0
        right[1] = 0.0
        forward_norm = float(np.linalg.norm(forward))
        right_norm = float(np.linalg.norm(right))
        if forward_norm > 0.0:
            forward /= forward_norm + 1e-6
        if right_norm > 0.0:
            right /= right_norm + 1e-6

        move = np.zeros(3, dtype=np.float64)
        if "w" in held:
            move += forward * translation_speed
        if "s" in held:
            move -= forward * translation_speed
        if "d" in held:
            move += right * translation_speed
        if "a" in held:
            move -= right * translation_speed

        current = np.eye(4, dtype=np.float64)
        current[:3, :3] = rotation
        current[:3, 3] = translation + move
        poses.append(current.copy())

    return np.stack(poses, axis=0).astype(np.float32)


def load_camera(path: Path) -> np.ndarray:
    c2w = np.load(path).astype(np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(
            f"camera trajectory must have shape (F, 4, 4); got {c2w.shape}"
        )
    return c2w


def load_intrinsics(path: Path, num_frames: int) -> np.ndarray:
    """Load intrinsics as ``(num_frames, 4)`` [fx, fy, cx, cy]."""
    arr = np.load(path).astype(np.float32)
    if arr.shape == (4,):
        return np.broadcast_to(arr, (num_frames, 4)).copy()
    if arr.shape == (3, 3):
        vec = np.array([arr[0, 0], arr[1, 1], arr[0, 2], arr[1, 2]], dtype=np.float32)
        return np.broadcast_to(vec, (num_frames, 4)).copy()
    if arr.ndim == 3 and arr.shape[1:] == (3, 3) and arr.shape[0] >= num_frames:
        arr = arr[:num_frames]
        return np.stack(
            [arr[:, 0, 0], arr[:, 1, 1], arr[:, 0, 2], arr[:, 1, 2]], axis=1
        )
    raise ValueError(
        f"unsupported intrinsics shape {arr.shape}; expected (4,), (3,3), or (F,3,3)"
    )


def snap_num_frames(
    num_frames: int, stride: int = 8, upper_bound: int | None = None
) -> int:
    """Snap to the nearest valid LTX-2 VAE length: ``stride * k + 1``."""
    if num_frames < 1:
        return 1
    if (num_frames - 1) % stride == 0:
        return min(num_frames, upper_bound) if upper_bound is not None else num_frames

    floor_candidate = num_frames - ((num_frames - 1) % stride)
    ceil_candidate = floor_candidate + stride
    snapped = (
        floor_candidate
        if num_frames - floor_candidate < ceil_candidate - num_frames
        else ceil_candidate
    )
    if upper_bound is not None and snapped > upper_bound:
        snapped = floor_candidate
    return max(snapped, 1)


def resize_and_center_crop(
    image: Image.Image,
    target_h: int = TARGET_HEIGHT,
    target_w: int = TARGET_WIDTH,
) -> tuple[Image.Image, tuple[int, int], tuple[int, int], tuple[int, int]]:
    src_w, src_h = image.size
    resized_w, resized_h, left, top = compute_resize_crop_geometry(
        src_w, src_h, target_h, target_w
    )
    resized = image.resize((resized_w, resized_h), Image.LANCZOS)
    crop = resized.crop((left, top, left + target_w, top + target_h))
    return crop, (src_w, src_h), (resized_w, resized_h), (left, top)


# Intrinsics crop transform and the raymap/chunk_plucker builders live solely in
# SanaWMBeforeDenoisingStage now: the realtime stage shares those helpers + compute_chunk_plucker
# with the batch path. A parallel impl agreed only to ~1-2 bf16 ulps and broke bitwise parity.
