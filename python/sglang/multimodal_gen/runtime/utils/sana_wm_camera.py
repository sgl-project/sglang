# SPDX-License-Identifier: Apache-2.0
"""SANA-WM camera trajectory and conditioning utilities."""

from __future__ import annotations

import math
import os
from typing import Any

import torch

SANA_WM_DEFAULT_TRANSLATION_SPEED = 0.05
SANA_WM_DEFAULT_ROTATION_SPEED_DEG = 1.2
SANA_WM_DEFAULT_PITCH_LIMIT_DEG = 85.0
SANA_WM_DEFAULT_HORIZONTAL_FOV_DEG = 70.0
SANA_WM_MIN_DEFAULT_FOV_DEG = 25.0
SANA_WM_MAX_DEFAULT_FOV_DEG = 120.0
SANA_WM_DEFAULT_HORIZONTAL_FOV_ENV = "SGLANG_SANA_WM_DEFAULT_HORIZONTAL_FOV_DEG"
SANA_WM_ALLOWED_ACTION_KEYS: frozenset[str] = frozenset("wasdijkl")


def validate_sana_wm_motion_params(
    *,
    translation_speed: Any = SANA_WM_DEFAULT_TRANSLATION_SPEED,
    rotation_speed_deg: Any = SANA_WM_DEFAULT_ROTATION_SPEED_DEG,
    pitch_limit_deg: Any = SANA_WM_DEFAULT_PITCH_LIMIT_DEG,
) -> tuple[float, float, float]:
    """Validate and normalize SANA-WM action rollout speed controls."""
    try:
        translation_speed = float(translation_speed)
        rotation_speed_deg = float(rotation_speed_deg)
        pitch_limit_deg = float(pitch_limit_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SANA-WM motion parameters must be numeric: "
            f"translation_speed={translation_speed!r}, "
            f"rotation_speed_deg={rotation_speed_deg!r}, "
            f"pitch_limit_deg={pitch_limit_deg!r}."
        ) from exc

    if not math.isfinite(translation_speed) or translation_speed <= 0:
        raise ValueError(
            "translation_speed must be a finite positive value, "
            f"got {translation_speed}."
        )
    if not math.isfinite(rotation_speed_deg) or not (0 < rotation_speed_deg < 180):
        raise ValueError(
            "rotation_speed_deg must be finite and in (0, 180) degrees, "
            f"got {rotation_speed_deg}."
        )
    if not math.isfinite(pitch_limit_deg) or not (0 < pitch_limit_deg <= 90):
        raise ValueError(
            "pitch_limit_deg must be finite and in (0, 90] degrees, "
            f"got {pitch_limit_deg}."
        )
    return translation_speed, rotation_speed_deg, pitch_limit_deg


def _sana_wm_rot_x(angle_rad: float) -> torch.Tensor:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64
    )


def _sana_wm_rot_y(angle_rad: float) -> torch.Tensor:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64
    )


def parse_sana_wm_action_string(action: str) -> list[list[str]]:
    """Parse upstream SANA-WM WASD/IJKL DSL into per-frame held keys."""

    cleaned = "".join(action.replace("，", ",").split())
    if not cleaned:
        raise ValueError("SANA-WM action string is empty")

    per_frame: list[list[str]] = []
    for segment in cleaned.split(","):
        if not segment or "-" not in segment:
            raise ValueError(
                f"Invalid SANA-WM action segment {segment!r}: "
                "expected '<keys>-<duration>'."
            )
        keys_part, dur_str = segment.rsplit("-", 1)
        if not dur_str.isdigit() or int(dur_str) <= 0:
            raise ValueError(
                f"SANA-WM action segment {segment!r} has a non-positive "
                f"duration {dur_str!r}."
            )

        keys_lower = keys_part.lower()
        if keys_lower == "none":
            keys: list[str] = []
        else:
            bad = sorted(
                {c for c in keys_lower if c not in SANA_WM_ALLOWED_ACTION_KEYS}
            )
            if bad:
                raise ValueError(
                    f"SANA-WM action segment {segment!r} contains unknown keys "
                    f"{bad}; allowed: {''.join(sorted(SANA_WM_ALLOWED_ACTION_KEYS))}."
                )
            keys = sorted(set(keys_lower))
        per_frame.extend([list(keys) for _ in range(int(dur_str))])
    return per_frame


def sana_wm_action_to_camera_to_world(
    action: str,
    *,
    translation_speed: float = SANA_WM_DEFAULT_TRANSLATION_SPEED,
    rotation_speed_deg: float = SANA_WM_DEFAULT_ROTATION_SPEED_DEG,
    pitch_limit_deg: float = SANA_WM_DEFAULT_PITCH_LIMIT_DEG,
) -> torch.Tensor:
    translation_speed, rotation_speed_deg, pitch_limit_deg = (
        validate_sana_wm_motion_params(
            translation_speed=translation_speed,
            rotation_speed_deg=rotation_speed_deg,
            pitch_limit_deg=pitch_limit_deg,
        )
    )
    per_frame = parse_sana_wm_action_string(action)
    rotate_rad = math.radians(rotation_speed_deg)
    pitch_limit_rad = math.radians(pitch_limit_deg)
    current = torch.eye(4, dtype=torch.float64)
    poses = [current.clone()]
    current_pitch = 0.0

    for keys in per_frame:
        held = set(keys)
        rotation = current[:3, :3]
        translation = current[:3, 3]

        pitch_delta = (rotate_rad if "i" in held else 0.0) - (
            rotate_rad if "k" in held else 0.0
        )
        new_pitch = current_pitch + pitch_delta
        if not (-pitch_limit_rad <= new_pitch <= pitch_limit_rad):
            pitch_delta = 0.0
        else:
            current_pitch = new_pitch

        yaw_delta = (rotate_rad if "l" in held else 0.0) - (
            rotate_rad if "j" in held else 0.0
        )
        rotation_new = (
            _sana_wm_rot_y(yaw_delta) @ rotation @ _sana_wm_rot_x(pitch_delta)
        )

        forward = rotation_new[:, 2].clone()
        forward[1] = 0.0
        right = rotation_new[:, 0].clone()
        right[1] = 0.0
        forward_norm = float(torch.linalg.vector_norm(forward).item())
        right_norm = float(torch.linalg.vector_norm(right).item())
        if forward_norm > 0:
            forward = forward / (forward_norm + 1e-6)
        if right_norm > 0:
            right = right / (right_norm + 1e-6)

        move = torch.zeros(3, dtype=torch.float64)
        if "w" in held:
            move += forward * translation_speed
        if "s" in held:
            move -= forward * translation_speed
        if "d" in held:
            move += right * translation_speed
        if "a" in held:
            move -= right * translation_speed

        current = torch.eye(4, dtype=torch.float64)
        current[:3, :3] = rotation_new
        current[:3, 3] = translation + move
        poses.append(current.clone())

    return torch.stack(poses, dim=0).to(dtype=torch.float32)


def sana_wm_action_num_frames(action: Any) -> int:
    if isinstance(action, str):
        return len(parse_sana_wm_action_string(action)) + 1
    if isinstance(action, (list, tuple)) and all(isinstance(x, str) for x in action):
        if len(action) == 0:
            raise ValueError("SANA-WM action list must not be empty.")
        return max(len(parse_sana_wm_action_string(x)) + 1 for x in action)
    raise ValueError(
        "SANA-WM action must be a string or a list of strings, "
        f"got {type(action).__name__}."
    )


def pad_or_trim_sana_wm_frames(tensor: torch.Tensor, num_frames: int) -> torch.Tensor:
    current = tensor.shape[1]
    if current == num_frames:
        return tensor
    if current > num_frames:
        return tensor[:, :num_frames]
    if current == 0:
        raise ValueError("camera trajectory must contain at least one frame")
    pad = num_frames - current
    last = tensor[:, -1:].repeat(1, pad, *([1] * (tensor.ndim - 2)))
    return torch.cat([tensor, last], dim=1)


def _pad_or_trim_sana_wm_action_trajectory(
    trajectory: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    current = trajectory.shape[0]
    if current == num_frames:
        return trajectory
    if current > num_frames:
        return trajectory[:num_frames]
    pad = num_frames - current
    return torch.cat([trajectory, trajectory[-1:].repeat(pad, 1, 1)], dim=0)


def _maybe_load_npy_tensor(value: Any, field_name: str) -> Any:
    if isinstance(value, (str, os.PathLike)):
        import numpy as np

        path = os.fspath(value)
        if not path.endswith(".npy"):
            raise ValueError(
                f"{field_name} path must point to a .npy file, got {path!r}"
            )
        return torch.from_numpy(np.load(path))
    return value


def coerce_sana_wm_action_camera_to_world(
    value: Any,
    *,
    batch_size: int,
    num_frames: int,
    translation_speed: float,
    rotation_speed_deg: float,
    pitch_limit_deg: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, str):
        actions = [value]
    elif isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value):
        actions = list(value)
    else:
        raise ValueError(
            "SANA-WM action must be a string or a list of strings, "
            f"got {type(value).__name__}."
        )

    trajectories = [
        _pad_or_trim_sana_wm_action_trajectory(
            sana_wm_action_to_camera_to_world(
                action,
                translation_speed=translation_speed,
                rotation_speed_deg=rotation_speed_deg,
                pitch_limit_deg=pitch_limit_deg,
            ),
            num_frames,
        )
        for action in actions
    ]
    camera = torch.stack(trajectories, dim=0).to(device=device, dtype=dtype)
    if camera.shape[0] == 1 and batch_size > 1:
        camera = camera.expand(batch_size, -1, -1, -1)
    elif camera.shape[0] != batch_size:
        raise ValueError(
            f"SANA-WM action batch {camera.shape[0]} does not match {batch_size}"
        )
    return camera


def coerce_sana_wm_camera_to_world(
    value: Any,
    *,
    batch_size: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = _maybe_load_npy_tensor(value, "camera_to_world")
    camera = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if camera.dim() == 3:
        camera = camera.unsqueeze(0)
    if camera.dim() != 4 or camera.shape[-2:] != (4, 4):
        raise ValueError(
            "camera_to_world must have shape (F,4,4) or (B,F,4,4), "
            f"got {tuple(camera.shape)}"
        )
    camera = camera.to(device=device, dtype=dtype)
    if camera.shape[0] == 1 and batch_size > 1:
        camera = camera.expand(batch_size, -1, -1, -1)
    elif camera.shape[0] != batch_size:
        raise ValueError(
            f"camera_to_world batch {camera.shape[0]} does not match {batch_size}"
        )
    return pad_or_trim_sana_wm_frames(camera, num_frames)


def sana_wm_intrinsics_matrix_to_vec4(intrinsics: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            intrinsics[..., 0, 0],
            intrinsics[..., 1, 1],
            intrinsics[..., 0, 2],
            intrinsics[..., 1, 2],
        ],
        dim=-1,
    )


def coerce_sana_wm_intrinsics_vec4(
    value: Any,
    *,
    batch_size: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = _maybe_load_npy_tensor(value, "intrinsics")
    intrinsics = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    intrinsics = intrinsics.to(device=device, dtype=dtype)

    if intrinsics.dim() == 1 and intrinsics.shape[0] == 4:
        intrinsics = intrinsics.view(1, 1, 4)
    elif intrinsics.dim() == 2 and intrinsics.shape == (3, 3):
        intrinsics = sana_wm_intrinsics_matrix_to_vec4(intrinsics).view(1, 1, 4)
    elif intrinsics.dim() == 2 and intrinsics.shape[-1] == 4:
        intrinsics = intrinsics.unsqueeze(0)
    elif intrinsics.dim() == 3 and intrinsics.shape[-2:] == (3, 3):
        vec4 = sana_wm_intrinsics_matrix_to_vec4(intrinsics)
        N = intrinsics.shape[0]
        if N == 1:
            # Single matrix: broadcast over both batch and frame axes.
            intrinsics = vec4.unsqueeze(0)  # (1, 1, 4)
        elif N == batch_size and N < num_frames:
            # Unambiguous: one matrix per batch item, same for all frames.
            intrinsics = vec4.unsqueeze(1)  # (B, 1, 4)
        elif N >= num_frames and N != batch_size:
            # Unambiguous: one matrix per frame (or more, trimmed later), broadcast batch.
            intrinsics = vec4.unsqueeze(0)  # (1, F, 4)
        elif N == batch_size == num_frames:
            # Ambiguous: N matches both batch_size and num_frames.
            # Treat as per-frame (broadcast batch) — the more common case
            # (camera intrinsics rarely differ per video in a batch).
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "intrinsics shape (N=%d, 3, 3) is ambiguous: N equals both "
                "batch_size and num_frames. Treating as per-frame intrinsics "
                "(broadcasting over batch). Pass shape (B, F, 3, 3) to be explicit.",
                N,
            )
            intrinsics = vec4.unsqueeze(0)  # (1, F, 4)
        else:
            raise ValueError(
                "intrinsics with shape (N, 3, 3) requires N == 1, "
                f"N == batch_size ({batch_size}), or N >= num_frames ({num_frames}); "
                f"got N={N}."
            )
    elif intrinsics.dim() == 3 and intrinsics.shape[-1] == 4:
        pass
    elif intrinsics.dim() == 4 and intrinsics.shape[-2:] == (3, 3):
        intrinsics = sana_wm_intrinsics_matrix_to_vec4(intrinsics)
    else:
        raise ValueError(
            "intrinsics must have shape (4,), (F,4), (B,F,4), "
            "(3,3), (F,3,3), or (B,F,3,3); "
            f"got {tuple(intrinsics.shape)}"
        )

    if intrinsics.shape[0] == 1 and batch_size > 1:
        intrinsics = intrinsics.expand(batch_size, -1, -1)
    elif intrinsics.shape[0] != batch_size:
        raise ValueError(
            f"intrinsics batch {intrinsics.shape[0]} does not match {batch_size}"
        )
    if intrinsics.shape[1] == 1 and num_frames > 1:
        intrinsics = intrinsics.expand(-1, num_frames, -1)
    return pad_or_trim_sana_wm_frames(intrinsics, num_frames)


def relative_sana_wm_camera_poses(camera_to_world: torch.Tensor) -> torch.Tensor:
    input_dtype = camera_to_world.dtype
    camera_to_world = camera_to_world.float()
    first_inv = torch.linalg.inv(camera_to_world[:, :1])
    poses = torch.matmul(first_inv, camera_to_world)
    eye = torch.eye(
        4,
        device=camera_to_world.device,
        dtype=camera_to_world.dtype,
    )
    poses[:, 0] = eye
    return poses.to(dtype=input_dtype)


def scale_sana_wm_intrinsics_to_latent(
    intrinsics_vec4: torch.Tensor,
    *,
    pixel_h: int,
    pixel_w: int,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    intrinsics_latent = intrinsics_vec4.clone()
    intrinsics_latent[..., [0, 2]] *= latent_w / float(pixel_w)
    intrinsics_latent[..., [1, 3]] *= latent_h / float(pixel_h)
    return intrinsics_latent


def flatten_sana_wm_camera_conditions(
    camera_to_world: torch.Tensor,
    intrinsics_vec4: torch.Tensor,
) -> torch.Tensor:
    c2w_flat = camera_to_world.reshape(
        camera_to_world.shape[0],
        camera_to_world.shape[1],
        16,
    )
    return torch.cat([c2w_flat, intrinsics_vec4], dim=-1)


def latent_frame_sana_wm_camera_conditions(
    camera_conditions: torch.Tensor,
    *,
    num_frames: int,
    latent_frames: int,
    vae_temporal_stride: int,
) -> torch.Tensor:
    time_indices = torch.arange(
        0,
        num_frames,
        vae_temporal_stride,
        device=camera_conditions.device,
        dtype=torch.long,
    )
    if time_indices.numel() < latent_frames:
        pad = latent_frames - int(time_indices.numel())
        time_indices = torch.cat([time_indices, time_indices[-1:].repeat(pad)], dim=0)
    time_indices = time_indices[:latent_frames]
    return camera_conditions.index_select(1, time_indices)


def sana_wm_default_horizontal_fov_deg() -> float:
    value = os.getenv(SANA_WM_DEFAULT_HORIZONTAL_FOV_ENV)
    if value is None or value.strip() == "":
        fov_deg = SANA_WM_DEFAULT_HORIZONTAL_FOV_DEG
    else:
        try:
            fov_deg = float(value)
        except ValueError as exc:
            raise ValueError(
                f"{SANA_WM_DEFAULT_HORIZONTAL_FOV_ENV} must be a float in "
                f"({SANA_WM_MIN_DEFAULT_FOV_DEG}, "
                f"{SANA_WM_MAX_DEFAULT_FOV_DEG}) degrees, got {value!r}."
            ) from exc

    if not (SANA_WM_MIN_DEFAULT_FOV_DEG < fov_deg < SANA_WM_MAX_DEFAULT_FOV_DEG):
        raise ValueError(
            f"SANA-WM default horizontal FOV must be in "
            f"({SANA_WM_MIN_DEFAULT_FOV_DEG}, "
            f"{SANA_WM_MAX_DEFAULT_FOV_DEG}) degrees, got {fov_deg}."
        )
    return fov_deg


def default_sana_wm_intrinsics_vec4(
    *,
    batch_size: int,
    num_frames: int,
    pixel_h: int,
    pixel_w: int,
    device: torch.device,
    dtype: torch.dtype,
    horizontal_fov_deg: float | None = None,
) -> torch.Tensor:
    if pixel_h <= 0 or pixel_w <= 0:
        raise ValueError(
            "SANA-WM default intrinsics require positive pixel size, "
            f"got height={pixel_h}, width={pixel_w}."
        )
    if horizontal_fov_deg is None:
        horizontal_fov_deg = sana_wm_default_horizontal_fov_deg()
    if not (
        SANA_WM_MIN_DEFAULT_FOV_DEG
        < float(horizontal_fov_deg)
        < SANA_WM_MAX_DEFAULT_FOV_DEG
    ):
        raise ValueError(
            f"SANA-WM default horizontal FOV must be in "
            f"({SANA_WM_MIN_DEFAULT_FOV_DEG}, "
            f"{SANA_WM_MAX_DEFAULT_FOV_DEG}) degrees, got "
            f"{horizontal_fov_deg}."
        )

    fov_rad = math.radians(float(horizontal_fov_deg))
    focal = float(pixel_w) / (2.0 * math.tan(fov_rad / 2.0))
    intrinsics = torch.tensor(
        [focal, focal, pixel_w / 2.0, pixel_h / 2.0],
        device=device,
        dtype=dtype,
    ).view(1, 1, 4)
    return intrinsics.repeat(batch_size, num_frames, 1)


def default_sana_wm_static_camera(
    *,
    batch_size: int,
    num_frames: int,
    pixel_h: int,
    pixel_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera_to_world = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4)
    camera_to_world = camera_to_world.repeat(batch_size, num_frames, 1, 1)
    intrinsics = default_sana_wm_intrinsics_vec4(
        batch_size=batch_size,
        num_frames=num_frames,
        pixel_h=pixel_h,
        pixel_w=pixel_w,
        device=device,
        dtype=dtype,
    )
    return camera_to_world, intrinsics
