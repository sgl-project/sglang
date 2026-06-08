# SPDX-License-Identifier: Apache-2.0

import math
import os
import time
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor

from sglang.jit_kernel.nvfp4 import prewarm_nvfp4_jit_modules
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    cfg_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.realtime.causal_state import RealtimeCausalDiTState
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE
from sglang.srt.utils.common import get_compiler_backend

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# SANA-WM stage-local helpers
# ---------------------------------------------------------------------------

SANA_WM_REQUEST_RUNTIME_CACHE_NAMESPACE = "sana_wm"
SANA_WM_DEFAULT_TRANSLATION_SPEED = 0.05
SANA_WM_DEFAULT_ROTATION_SPEED_DEG = 1.2
SANA_WM_DEFAULT_PITCH_LIMIT_DEG = 85.0
SANA_WM_DEFAULT_HORIZONTAL_FOV_DEG = 70.0
SANA_WM_MIN_DEFAULT_FOV_DEG = 25.0
SANA_WM_MAX_DEFAULT_FOV_DEG = 120.0
SANA_WM_ALLOWED_ACTION_KEYS: frozenset[str] = frozenset("wasdijkl")
SANA_WM_CAMERA_SOURCE_KEYS: frozenset[str] = frozenset(
    ("action", "camera_to_world", "camera_conditions")
)
SANA_WM_CAMERA_MOTION_KEYS: frozenset[str] = frozenset(
    ("translation_speed", "rotation_speed_deg", "pitch_limit_deg")
)


def _clear_sana_wm_request_runtime_cache(batch: Any) -> None:
    session = getattr(batch, "session", None)
    if session is not None:
        get_state = getattr(session, "get_state", None)
        state = get_state(RealtimeCausalDiTState) if callable(get_state) else None
        if state is not None:
            state.runtime_cache.pop(SANA_WM_REQUEST_RUNTIME_CACHE_NAMESPACE, None)

    extra = getattr(batch, "extra", None)
    if extra is not None:
        extra.pop(SANA_WM_REQUEST_RUNTIME_CACHE_NAMESPACE, None)


def sana_wm_default_horizontal_fov_deg(value=None) -> float:
    if value is None:
        fov_deg = SANA_WM_DEFAULT_HORIZONTAL_FOV_DEG
    else:
        try:
            fov_deg = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SANA-WM default horizontal FOV must be a number, " f"got {value!r}."
            ) from exc
    if not SANA_WM_MIN_DEFAULT_FOV_DEG < fov_deg < SANA_WM_MAX_DEFAULT_FOV_DEG:
        raise ValueError(
            f"SANA-WM default horizontal FOV must be in "
            f"({SANA_WM_MIN_DEFAULT_FOV_DEG}, "
            f"{SANA_WM_MAX_DEFAULT_FOV_DEG}) degrees, got {fov_deg}."
        )
    return fov_deg


def parse_sana_wm_action_string(action: str) -> list[list[str]]:
    cleaned = "".join(str(action).replace("，", ",").split())
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
            invalid = sorted(set(keys_lower) - SANA_WM_ALLOWED_ACTION_KEYS)
            if invalid:
                raise ValueError(
                    f"SANA-WM action segment {segment!r} contains unsupported keys "
                    f"{invalid}; allowed keys are "
                    f"{sorted(SANA_WM_ALLOWED_ACTION_KEYS)}."
                )
            keys = sorted(set(keys_lower))
        per_frame.extend([list(keys) for _ in range(int(dur_str))])
    return per_frame


def validate_sana_wm_motion_params(
    *,
    translation_speed=SANA_WM_DEFAULT_TRANSLATION_SPEED,
    rotation_speed_deg=SANA_WM_DEFAULT_ROTATION_SPEED_DEG,
    pitch_limit_deg=SANA_WM_DEFAULT_PITCH_LIMIT_DEG,
) -> tuple[float, float, float]:
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


def normalize_sana_wm_condition_inputs(condition_inputs: Any = None) -> dict[str, Any]:
    allowed_keys = (
        SANA_WM_CAMERA_SOURCE_KEYS
        | SANA_WM_CAMERA_MOTION_KEYS
        | frozenset(("chunk_plucker", "intrinsics"))
    )
    if condition_inputs is None:
        normalized: dict[str, Any] = {}
    elif not isinstance(condition_inputs, dict):
        raise ValueError(
            "SANA-WM condition_inputs must be a dict, "
            f"got {type(condition_inputs).__name__}."
        )
    else:
        normalized = {
            key: value for key, value in condition_inputs.items() if value is not None
        }

    unknown_keys = sorted(set(normalized) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            "SANA-WM condition_inputs contains unknown keys "
            f"{unknown_keys}; allowed keys are {sorted(allowed_keys)}."
        )

    present = {key for key, value in normalized.items() if value is not None}
    source_keys = sorted(present & SANA_WM_CAMERA_SOURCE_KEYS)
    if len(source_keys) > 1:
        raise ValueError(
            "SANA-WM camera source inputs are mutually exclusive; got "
            f"{source_keys}. Use exactly one of action, camera_to_world, or "
            "camera_conditions. chunk_plucker is allowed only as an auxiliary "
            "conditioning tensor."
        )
    if "action" in present and "chunk_plucker" in present:
        raise ValueError(
            "SANA-WM action is mutually exclusive with chunk_plucker because "
            "the action trajectory derives its own Plucker conditioning."
        )
    if "camera_conditions" in present and "intrinsics" in present:
        raise ValueError(
            "SANA-WM camera_conditions already contains flattened camera pose "
            "and intrinsics, so it cannot be combined with intrinsics."
        )
    if (present & SANA_WM_CAMERA_MOTION_KEYS) and "action" not in present:
        raise ValueError(
            "SANA-WM motion parameters "
            f"{sorted(present & SANA_WM_CAMERA_MOTION_KEYS)} require action."
        )

    if "action" in present:
        parse_sana_wm_action_string(normalized["action"])
        (
            normalized["translation_speed"],
            normalized["rotation_speed_deg"],
            normalized["pitch_limit_deg"],
        ) = validate_sana_wm_motion_params(
            translation_speed=normalized.get(
                "translation_speed", SANA_WM_DEFAULT_TRANSLATION_SPEED
            ),
            rotation_speed_deg=normalized.get(
                "rotation_speed_deg", SANA_WM_DEFAULT_ROTATION_SPEED_DEG
            ),
            pitch_limit_deg=normalized.get(
                "pitch_limit_deg", SANA_WM_DEFAULT_PITCH_LIMIT_DEG
            ),
        )
    return normalized


def sana_wm_condition_inputs_are_valid(condition_inputs: Any) -> bool:
    try:
        normalize_sana_wm_condition_inputs(condition_inputs)
    except (TypeError, ValueError):
        return False
    return True


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
    horizontal_fov_deg = sana_wm_default_horizontal_fov_deg(horizontal_fov_deg)

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
    horizontal_fov_deg: float | None = None,
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
        horizontal_fov_deg=horizontal_fov_deg,
    )
    return camera_to_world, intrinsics


SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY = "sana_wm_condition_image_preprocess"

SanaWMConditionImagePreprocessInfo = dict[str, tuple[int, int]]

_SANA_WM_CHANNEL_COUNTS = {1, 3, 4}


def canonical_sana_wm_condition_image_tensor(image: torch.Tensor) -> torch.Tensor:
    """Return image as NCHW RGB float tensor without changing its value range.

    Accepted input layouts:
      - NCHW  (N, 1|3|4, H, W)
      - NHWC  (N, H, W, 1|3|4)
      - CHW   (1|3|4, H, W) when the channel dim is unambiguous
      - HWC   (H, W, 1|3|4) when the channel dim is unambiguous
      - NCFHW singleton-video (N, C, 1, H, W) squeezed to NCHW
    """
    image = image.float()
    if image.dim() == 5 and image.shape[2] == 1:
        image = image.squeeze(2)
    if image.dim() == 3:
        c_first, c_last = image.shape[0], image.shape[-1]
        is_channel_first = c_first in _SANA_WM_CHANNEL_COUNTS
        is_channel_last = c_last in _SANA_WM_CHANNEL_COUNTS
        if is_channel_first and not is_channel_last:
            image = image.unsqueeze(0)
        elif is_channel_last and not is_channel_first:
            image = image.permute(2, 0, 1).unsqueeze(0)
        elif is_channel_first and is_channel_last:
            raise ValueError(
                f"Ambiguous condition_image tensor shape {tuple(image.shape)}: "
                f"both leading dim ({c_first}) and trailing dim ({c_last}) look "
                "like channel counts (1, 3, or 4). Pass an explicit NCHW tensor."
            )
        else:
            raise ValueError(
                "condition_image tensor must be CHW or HWC with 1, 3, or 4 "
                f"channels; got shape {tuple(image.shape)}."
            )
    elif image.dim() == 4:
        if image.shape[1] in _SANA_WM_CHANNEL_COUNTS:
            pass
        elif image.shape[-1] in _SANA_WM_CHANNEL_COUNTS:
            image = image.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "condition_image tensor must be NCHW or NHWC with 1, 3, "
                f"or 4 channels; got {tuple(image.shape)}."
            )
    else:
        raise ValueError(
            "condition_image tensor must have shape CHW, HWC, NCHW, NHWC, "
            f"or NCHW singleton-video; got {tuple(image.shape)}."
        )

    if image.shape[1] == 1:
        image = image.expand(-1, 3, -1, -1)
    elif image.shape[1] == 4:
        image = image[:, :3]
    elif image.shape[1] != 3:
        raise ValueError(
            f"condition_image must have 1, 3, or 4 channels; got {image.shape[1]}."
        )
    return image.contiguous()


def resize_center_crop_sana_wm_pil_image(
    image: Any,
    *,
    target_h: int,
    target_w: int,
) -> tuple[torch.Tensor, SanaWMConditionImagePreprocessInfo]:
    """Resize then center-crop SANA-WM first-frame conditioning images."""
    import PIL.Image
    import torchvision.transforms.functional as TF

    image = image.convert("RGB")
    src_w, src_h = image.size
    scale = max(target_h / float(src_h), target_w / float(src_w))
    resized_w = max(target_w, int(round(src_w * scale)))
    resized_h = max(target_h, int(round(src_h * scale)))
    resampling_enum = getattr(PIL.Image, "Resampling", None)
    resampling = (
        resampling_enum.LANCZOS if resampling_enum is not None else PIL.Image.LANCZOS
    )
    image = image.resize((resized_w, resized_h), resampling)
    left = (resized_w - target_w) // 2
    top = (resized_h - target_h) // 2
    image = image.crop((left, top, left + target_w, top + target_h))
    return TF.to_tensor(image).unsqueeze(0), {
        "source_size": (src_w, src_h),
        "resized_size": (resized_w, resized_h),
        "crop_offset": (left, top),
        "target_size": (target_w, target_h),
    }


def sana_wm_condition_tensor_to_pil_images(image: torch.Tensor) -> list[Any]:
    """Convert tensor inputs to RGB PIL images before preprocessing."""
    import PIL.Image

    image = canonical_sana_wm_condition_image_tensor(image)
    pil_images = []
    for sample in image.detach().cpu():
        sample = sample.float()
        if sample.max() > 1.5:
            sample = sample / 255.0
        elif sample.min() < 0.0:
            sample = (sample + 1.0) * 0.5
        sample = sample.clamp(0.0, 1.0)
        sample_u8 = (
            sample.mul(255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous()
        )
        pil_images.append(PIL.Image.fromarray(sample_u8.numpy()))
    return pil_images


def resize_center_crop_sana_wm_tensor(
    image: torch.Tensor,
    *,
    target_h: int,
    target_w: int,
) -> tuple[torch.Tensor, SanaWMConditionImagePreprocessInfo]:
    """Route tensor inputs through the same PIL/LANCZOS path as upstream."""
    pil_images = sana_wm_condition_tensor_to_pil_images(image)
    resized = [
        resize_center_crop_sana_wm_pil_image(
            pil_image,
            target_h=target_h,
            target_w=target_w,
        )
        for pil_image in pil_images
    ]
    image_tensors = [item[0] for item in resized]
    preprocess_info = resized[0][1]
    return torch.cat(image_tensors, dim=0), preprocess_info


def preprocess_sana_wm_condition_image(
    condition_image: Any,
    *,
    target_h: int,
    target_w: int,
) -> tuple[torch.Tensor, SanaWMConditionImagePreprocessInfo]:
    """Aspect-preserving resize + center crop, mirroring NVlabs/Sana."""
    import PIL.Image

    if isinstance(condition_image, list):
        if len(condition_image) == 0:
            raise ValueError(
                "condition_image list is empty; SANA-WM requires a first "
                "frame conditioning image."
            )
        condition_image = condition_image[0]

    if isinstance(condition_image, PIL.Image.Image):
        return resize_center_crop_sana_wm_pil_image(
            condition_image,
            target_h=target_h,
            target_w=target_w,
        )

    if isinstance(condition_image, torch.Tensor):
        return resize_center_crop_sana_wm_tensor(
            condition_image,
            target_h=target_h,
            target_w=target_w,
        )

    raise TypeError(
        "condition_image must be a PIL image, tensor, or non-empty list; "
        f"got {type(condition_image).__name__}."
    )


def transform_sana_wm_intrinsics_for_condition_image(
    intrinsics_vec4: torch.Tensor,
    preprocess_info: (
        SanaWMConditionImagePreprocessInfo
        | list[SanaWMConditionImagePreprocessInfo]
        | None
    ),
) -> torch.Tensor:
    """Map source-image intrinsics into the cropped output pixel grid."""
    if not preprocess_info:
        return intrinsics_vec4
    if isinstance(preprocess_info, list):
        if len(preprocess_info) == 1:
            return transform_sana_wm_intrinsics_for_condition_image(
                intrinsics_vec4, preprocess_info[0]
            )
        if len(preprocess_info) != intrinsics_vec4.shape[0]:
            raise ValueError(
                "SANA-WM condition-image preprocess metadata length must "
                "match intrinsics batch size; got "
                f"{len(preprocess_info)} metadata entries for batch "
                f"{intrinsics_vec4.shape[0]}."
            )
        return torch.cat(
            [
                transform_sana_wm_intrinsics_for_condition_image(
                    intrinsics_vec4[index : index + 1], info
                )
                for index, info in enumerate(preprocess_info)
            ],
            dim=0,
        )
    src_w, src_h = preprocess_info["source_size"]
    resized_w, resized_h = preprocess_info["resized_size"]
    left, top = preprocess_info["crop_offset"]
    sx = resized_w / float(src_w)
    sy = resized_h / float(src_h)
    out = intrinsics_vec4.clone()
    out[..., 0] *= sx
    out[..., 2] = out[..., 2] * sx - left
    out[..., 1] *= sy
    out[..., 3] = out[..., 3] * sy - top
    return out


def set_sana_wm_condition_image_preprocess_info(
    batch: Any | None,
    preprocess_info: Any,
) -> None:
    if batch is None:
        return
    if not hasattr(batch, "extra") or batch.extra is None:
        batch.extra = {}
    batch.extra[SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY] = preprocess_info


def sana_wm_condition_images_for_batch(
    condition_image: Any,
    batch_size: int,
) -> list[Any]:
    if isinstance(condition_image, list):
        if not condition_image:
            raise ValueError(
                "condition_image list is empty; SANA-WM requires a first "
                "frame conditioning image."
            )
        if len(condition_image) == 1 or len(condition_image) == batch_size:
            return list(condition_image)
        raise ValueError(
            "SANA-WM condition_image list must contain one image or one "
            f"image per batch item; got {len(condition_image)} images for "
            f"batch {batch_size}."
        )
    return [condition_image]


def preprocess_sana_wm_condition_images_for_batch(
    condition_image: Any,
    *,
    batch_size: int,
    target_h: int,
    target_w: int,
) -> tuple[torch.Tensor, Any, list[SanaWMConditionImagePreprocessInfo]]:
    condition_images = sana_wm_condition_images_for_batch(condition_image, batch_size)
    first_frame_images = []
    preprocess_infos = []
    for image in condition_images:
        img_tensor, preprocess_info = preprocess_sana_wm_condition_image(
            image,
            target_h=target_h,
            target_w=target_w,
        )
        preprocess_infos.append(preprocess_info)
        first_frame_images.append(img_tensor)

    preprocess_info_for_batch = (
        preprocess_infos[0] if len(preprocess_infos) == 1 else preprocess_infos
    )
    return (
        torch.cat(first_frame_images, dim=0),
        preprocess_info_for_batch,
        preprocess_infos,
    )


def sana_wm_action_num_frames_for_request(batch: Any) -> int | None:
    action = normalize_sana_wm_condition_inputs(
        getattr(batch, "condition_inputs", None)
    ).get("action")
    if action is None:
        return None
    return sana_wm_action_num_frames(action)


class SanaWMCameraConditioningBuilder:
    """Build latent-frame raymap and Plucker conditioning for one request."""

    def __init__(
        self,
        pipeline_config: Any,
        *,
        log_info: Callable[..., None] | None = None,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.log_info = log_info or (lambda *args, **kwargs: None)

    def build(
        self,
        batch: Any,
        *,
        batch_size: int,
        num_frames: int,
        latent_shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
        if not hasattr(batch, "extra") or batch.extra is None:
            batch.extra = {}
        if not getattr(self.pipeline_config, "camera_conditioning", True):
            return None, None, "disabled"

        extra = batch.extra
        condition_inputs = normalize_sana_wm_condition_inputs(
            getattr(batch, "condition_inputs", None)
        )
        T_lat = latent_shape[2]
        sp_h = latent_shape[3]
        sp_w = latent_shape[4]
        vae_temporal_stride = self.pipeline_config.vae_stride[0]
        camera_compute_dtype = torch.float32

        camera_conditions = condition_inputs.get("camera_conditions")
        chunk_plucker = condition_inputs.get("chunk_plucker")
        preprocess_info = extra.get(SANA_WM_CONDITION_IMAGE_PREPROCESS_KEY)
        arch = getattr(
            getattr(self.pipeline_config, "dit_config", None),
            "arch_config",
            None,
        )
        default_horizontal_fov_deg = sana_wm_default_horizontal_fov_deg(
            getattr(self.pipeline_config, "sana_wm_default_horizontal_fov_deg", None)
        )
        requires_chunk_plucker = bool(
            getattr(arch, "use_chunk_plucker_post_attn", False)
            or getattr(arch, "use_chunk_plucker_input", False)
        )
        if camera_conditions is not None:
            camera_conditions, chunk_plucker, source = (
                self._build_from_camera_conditions(
                    camera_conditions,
                    chunk_plucker,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    latent_frames=T_lat,
                    sp_h=sp_h,
                    sp_w=sp_w,
                    vae_temporal_stride=vae_temporal_stride,
                    requires_chunk_plucker=requires_chunk_plucker,
                    device=device,
                    dtype=camera_compute_dtype,
                )
            )
        else:
            camera_conditions, chunk_plucker, source = self._build_from_camera_request(
                batch,
                condition_inputs,
                chunk_plucker,
                preprocess_info,
                batch_size=batch_size,
                num_frames=num_frames,
                latent_frames=T_lat,
                sp_h=sp_h,
                sp_w=sp_w,
                vae_temporal_stride=vae_temporal_stride,
                default_horizontal_fov_deg=default_horizontal_fov_deg,
                device=device,
                dtype=camera_compute_dtype,
            )

        chunk_plucker = self._coerce_chunk_plucker(
            chunk_plucker,
            batch_size=batch_size,
            latent_frames=T_lat,
            sp_h=sp_h,
            sp_w=sp_w,
            device=device,
            dtype=dtype,
        )
        if camera_conditions is not None:
            camera_conditions = camera_conditions.to(device=device, dtype=dtype)

        return camera_conditions, chunk_plucker, source

    def _build_from_camera_conditions(
        self,
        camera_conditions: Any,
        chunk_plucker: Any,
        *,
        batch_size: int,
        num_frames: int,
        latent_frames: int,
        sp_h: int,
        sp_w: int,
        vae_temporal_stride: int,
        requires_chunk_plucker: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Any, str]:
        camera_conditions = (
            camera_conditions
            if isinstance(camera_conditions, torch.Tensor)
            else torch.as_tensor(camera_conditions)
        ).to(device=device, dtype=dtype)
        if camera_conditions.dim() == 2:
            camera_conditions = camera_conditions.unsqueeze(0)
        if camera_conditions.dim() != 3:
            raise ValueError(
                "camera_conditions must have shape (T,20) or (B,T,20), "
                f"got {tuple(camera_conditions.shape)}"
            )
        if camera_conditions.shape[0] == 1 and batch_size > 1:
            camera_conditions = camera_conditions.expand(batch_size, -1, -1)
        if camera_conditions.shape[0] != batch_size:
            raise ValueError(
                "camera_conditions batch dimension must be 1 or match "
                f"request batch size {batch_size}, got "
                f"{camera_conditions.shape[0]}."
            )
        if camera_conditions.shape[-1] != 20:
            raise ValueError(
                "camera_conditions must have last dimension 20, got "
                f"{tuple(camera_conditions.shape)}"
            )
        if camera_conditions.shape[1] == latent_frames:
            source = "prepacked"
            if chunk_plucker is None and requires_chunk_plucker:
                raise ValueError(
                    "Prepacked latent-frame camera_conditions require "
                    "chunk_plucker for this SANA-WM checkpoint. Pass "
                    "chunk_plucker with shape (B,48,T,H,W), or pass "
                    "original-frame camera_conditions so SGLang can "
                    "derive chunk_plucker."
                )
            return camera_conditions, chunk_plucker, source

        source = "prebuilt_original_frames"
        original_camera_conditions = pad_or_trim_sana_wm_frames(
            camera_conditions, num_frames
        )
        camera_conditions = latent_frame_sana_wm_camera_conditions(
            original_camera_conditions,
            num_frames=num_frames,
            latent_frames=latent_frames,
            vae_temporal_stride=vae_temporal_stride,
        )
        if chunk_plucker is None:
            chunk_plucker = self._compute_chunk_plucker(
                original_camera_conditions,
                latent_frames=latent_frames,
                sp_h=sp_h,
                sp_w=sp_w,
                vae_temporal_stride=vae_temporal_stride,
            )
        return camera_conditions, chunk_plucker, source

    def _build_from_camera_request(
        self,
        batch: Any,
        condition_inputs: dict[str, Any],
        chunk_plucker: Any,
        preprocess_info: Any,
        *,
        batch_size: int,
        num_frames: int,
        latent_frames: int,
        sp_h: int,
        sp_w: int,
        vae_temporal_stride: int,
        default_horizontal_fov_deg: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Any, str]:
        camera_to_world = condition_inputs.get("camera_to_world")
        intrinsics = condition_inputs.get("intrinsics")
        action = condition_inputs.get("action")

        if action is not None:
            source = "action" if intrinsics is not None else "action_default_intrinsics"
            (
                camera_to_world,
                intrinsics_vec4,
            ) = self._build_action_camera_and_intrinsics(
                batch,
                condition_inputs,
                intrinsics,
                preprocess_info,
                batch_size=batch_size,
                num_frames=num_frames,
                default_horizontal_fov_deg=default_horizontal_fov_deg,
                device=device,
                dtype=dtype,
            )
        elif camera_to_world is not None:
            source = (
                "request" if intrinsics is not None else "request_default_intrinsics"
            )
            camera_to_world = coerce_sana_wm_camera_to_world(
                camera_to_world,
                batch_size=batch_size,
                num_frames=num_frames,
                device=device,
                dtype=dtype,
            )
            intrinsics_vec4 = self._coerce_or_default_intrinsics(
                batch,
                intrinsics,
                preprocess_info,
                batch_size=batch_size,
                num_frames=num_frames,
                default_horizontal_fov_deg=default_horizontal_fov_deg,
                device=device,
                dtype=dtype,
                log_default_context="request camera trajectory",
            )
        elif intrinsics is not None:
            source = "default_static_request_intrinsics"
            camera_to_world, _ = default_sana_wm_static_camera(
                batch_size=batch_size,
                num_frames=num_frames,
                pixel_h=batch.height,
                pixel_w=batch.width,
                device=device,
                dtype=dtype,
                horizontal_fov_deg=default_horizontal_fov_deg,
            )
            intrinsics_vec4 = coerce_sana_wm_intrinsics_vec4(
                intrinsics,
                batch_size=batch_size,
                num_frames=num_frames,
                device=device,
                dtype=dtype,
            )
            intrinsics_vec4 = transform_sana_wm_intrinsics_for_condition_image(
                intrinsics_vec4,
                preprocess_info,
            )
            self.log_info(
                "No camera trajectory provided; using static identity "
                "poses with request intrinsics."
            )
        else:
            source = "default_static"
            self.log_info(
                "No camera trajectory provided; using a static identity "
                "camera with centered pinhole intrinsics at default "
                "horizontal FOV %.2f deg. Pass camera_to_world/intrinsics "
                "for camera-controlled output.",
                default_horizontal_fov_deg,
            )
            camera_to_world, intrinsics_vec4 = default_sana_wm_static_camera(
                batch_size=batch_size,
                num_frames=num_frames,
                pixel_h=batch.height,
                pixel_w=batch.width,
                device=device,
                dtype=dtype,
                horizontal_fov_deg=default_horizontal_fov_deg,
            )

        camera_to_world = relative_sana_wm_camera_poses(camera_to_world)
        intrinsics_vec4 = scale_sana_wm_intrinsics_to_latent(
            intrinsics_vec4,
            pixel_h=batch.height,
            pixel_w=batch.width,
            latent_h=sp_h,
            latent_w=sp_w,
        )
        original_camera_conditions = flatten_sana_wm_camera_conditions(
            camera_to_world, intrinsics_vec4
        )
        camera_conditions = latent_frame_sana_wm_camera_conditions(
            original_camera_conditions,
            num_frames=num_frames,
            latent_frames=latent_frames,
            vae_temporal_stride=vae_temporal_stride,
        )
        if chunk_plucker is None:
            chunk_plucker = self._compute_chunk_plucker(
                original_camera_conditions,
                latent_frames=latent_frames,
                sp_h=sp_h,
                sp_w=sp_w,
                vae_temporal_stride=vae_temporal_stride,
            )
        return camera_conditions, chunk_plucker, source

    def _build_action_camera_and_intrinsics(
        self,
        batch: Any,
        condition_inputs: dict[str, Any],
        intrinsics: Any,
        preprocess_info: Any,
        *,
        batch_size: int,
        num_frames: int,
        default_horizontal_fov_deg: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        translation_speed = condition_inputs["translation_speed"]
        rotation_speed_deg = condition_inputs["rotation_speed_deg"]
        pitch_limit_deg = condition_inputs["pitch_limit_deg"]
        camera_to_world = coerce_sana_wm_action_camera_to_world(
            condition_inputs["action"],
            batch_size=batch_size,
            num_frames=num_frames,
            translation_speed=translation_speed,
            rotation_speed_deg=rotation_speed_deg,
            pitch_limit_deg=pitch_limit_deg,
            device=device,
            dtype=dtype,
        )
        self.log_info(
            "SANA-WM action trajectory rolled out: frames=%d, "
            "translation_speed=%.6g, rotation_speed_deg=%.6g, "
            "pitch_limit_deg=%.6g",
            camera_to_world.shape[1],
            translation_speed,
            rotation_speed_deg,
            pitch_limit_deg,
        )
        intrinsics_vec4 = self._coerce_or_default_intrinsics(
            batch,
            intrinsics,
            preprocess_info,
            batch_size=batch_size,
            num_frames=num_frames,
            default_horizontal_fov_deg=default_horizontal_fov_deg,
            device=device,
            dtype=dtype,
            log_default_context="action trajectory",
        )
        return camera_to_world, intrinsics_vec4

    def _coerce_or_default_intrinsics(
        self,
        batch: Any,
        intrinsics: Any,
        preprocess_info: Any,
        *,
        batch_size: int,
        num_frames: int,
        default_horizontal_fov_deg: float,
        device: torch.device,
        dtype: torch.dtype,
        log_default_context: str,
    ) -> torch.Tensor:
        if intrinsics is None:
            _, intrinsics_vec4 = default_sana_wm_static_camera(
                batch_size=batch_size,
                num_frames=num_frames,
                pixel_h=batch.height,
                pixel_w=batch.width,
                device=device,
                dtype=dtype,
                horizontal_fov_deg=default_horizontal_fov_deg,
            )
            self.log_info(
                "No intrinsics provided; using centered pinhole intrinsics "
                "with default horizontal FOV %.2f deg for the %s. Pass "
                "request intrinsics for camera-accurate geometry.",
                default_horizontal_fov_deg,
                log_default_context,
            )
            return intrinsics_vec4

        intrinsics_vec4 = coerce_sana_wm_intrinsics_vec4(
            intrinsics,
            batch_size=batch_size,
            num_frames=num_frames,
            device=device,
            dtype=dtype,
        )
        return transform_sana_wm_intrinsics_for_condition_image(
            intrinsics_vec4,
            preprocess_info,
        )

    @staticmethod
    def _compute_chunk_plucker(
        original_camera_conditions: torch.Tensor,
        *,
        latent_frames: int,
        sp_h: int,
        sp_w: int,
        vae_temporal_stride: int,
    ) -> torch.Tensor:
        B, frame_count, _ = original_camera_conditions.shape
        T, H, W = latent_frames, sp_h, sp_w
        device = original_camera_conditions.device
        dtype = original_camera_conditions.dtype

        c2w = original_camera_conditions[..., :16].view(B, frame_count, 4, 4)
        fx = original_camera_conditions[..., 16]
        fy = original_camera_conditions[..., 17]
        cx = original_camera_conditions[..., 18]
        cy = original_camera_conditions[..., 19]

        x_fov = 2.0 * torch.atan(W / (2.0 * fx.clamp(min=1e-6)))
        y_fov = 2.0 * torch.atan(H / (2.0 * fy.clamp(min=1e-6)))

        u = torch.arange(W, device=device, dtype=dtype)
        v = torch.arange(H, device=device, dtype=dtype)
        u = u.view(1, 1, 1, W).expand(B, frame_count, H, W)
        v = v.view(1, 1, H, 1).expand(B, frame_count, H, W)
        cx_e = cx.view(B, frame_count, 1, 1)
        cy_e = cy.view(B, frame_count, 1, 1)
        tan_x = torch.tan(x_fov / 2.0).view(B, frame_count, 1, 1)
        tan_y = torch.tan(y_fov / 2.0).view(B, frame_count, 1, 1)
        d_cam = torch.stack(
            [
                (u - cx_e) / max(W, 1) * 2.0 * tan_x,
                (v - cy_e) / max(H, 1) * 2.0 * tan_y,
                torch.ones_like(u),
            ],
            dim=-1,
        )
        d_cam = F.normalize(d_cam, dim=-1)

        R = c2w[..., :3, :3]
        origin = c2w[..., :3, 3]
        d_world = F.normalize(torch.einsum("bfij,bfhwj->bfhwi", R, d_cam), dim=-1)
        origin = origin.view(B, frame_count, 1, 1, 3).expand_as(d_world)
        moment = torch.cross(origin, d_world, dim=-1)
        plucker = torch.cat([d_world, moment], dim=-1)

        time_indices = list(range(0, frame_count, vae_temporal_stride))
        if len(time_indices) < T:
            last = time_indices[-1] if time_indices else 0
            time_indices.extend([last] * (T - len(time_indices)))
        time_indices = time_indices[:T]

        chunks = []
        for time_index in time_indices:
            start = max(0, int(time_index) - vae_temporal_stride + 1)
            end = min(start + vae_temporal_stride, frame_count)
            chunk = plucker[:, start:end]
            if chunk.shape[1] < vae_temporal_stride:
                pad = vae_temporal_stride - chunk.shape[1]
                chunk = torch.cat([chunk, chunk[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1)
            chunks.append(chunk)

        plucker = torch.stack(chunks, dim=1)
        plucker = plucker.permute(0, 1, 3, 4, 2, 5).reshape(
            B,
            T,
            H,
            W,
            vae_temporal_stride * 6,
        )
        return plucker.permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def _coerce_chunk_plucker(
        chunk_plucker: Any,
        *,
        batch_size: int,
        latent_frames: int,
        sp_h: int,
        sp_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if chunk_plucker is None:
            return None

        chunk_plucker = (
            chunk_plucker
            if isinstance(chunk_plucker, torch.Tensor)
            else torch.as_tensor(chunk_plucker)
        ).to(device=device, dtype=dtype)
        if chunk_plucker.dim() == 4:
            chunk_plucker = chunk_plucker.unsqueeze(0)
        if chunk_plucker.shape[0] == 1 and batch_size > 1:
            chunk_plucker = chunk_plucker.expand(batch_size, -1, -1, -1, -1)
        if chunk_plucker.dim() != 5:
            raise ValueError(
                "chunk_plucker must have shape (48,T,H,W) or "
                f"(B,48,T,H,W), got {tuple(chunk_plucker.shape)}"
            )
        if chunk_plucker.shape[0] != batch_size:
            raise ValueError(
                "chunk_plucker batch dimension must be 1 or match "
                f"request batch size {batch_size}, got "
                f"{chunk_plucker.shape[0]}."
            )
        expected_chunk_shape = (batch_size, 48, latent_frames, sp_h, sp_w)
        if tuple(chunk_plucker.shape) != expected_chunk_shape:
            raise ValueError(
                "chunk_plucker shape mismatch for SANA-WM: expected "
                f"{expected_chunk_shape}, got {tuple(chunk_plucker.shape)}."
            )
        return chunk_plucker


_SANA_WM_DIAGNOSTIC_MAX_EXACT_ELEMENTS = 4_194_304
_SANA_WM_DIAGNOSTIC_MAX_SAMPLE_ELEMENTS = 65_536

_SANA_WM_DEFAULT_VAE_TILE_MIN_FRAMES = 96
_SANA_WM_DEFAULT_VAE_TILE_STRIDE_FRAMES = 64


def sana_wm_diagnostics_enabled(pipeline_config: Any | None = None) -> bool:
    """Whether to emit detailed SANA-WM tensor-quality diagnostics."""
    return bool(getattr(pipeline_config, "sana_wm_diagnostics", False))


def log_sana_wm_tensor_stats(
    label: str,
    tensor: torch.Tensor | None,
    pipeline_config: Any | None = None,
) -> None:
    if not sana_wm_diagnostics_enabled(pipeline_config):
        return
    if tensor is None:
        logger.info("[SANA-WM diagnostics] %s: None", label)
        return
    if not isinstance(tensor, torch.Tensor):
        logger.info(
            "[SANA-WM diagnostics] %s: non-tensor type=%s",
            label,
            type(tensor).__name__,
        )
        return

    with torch.no_grad():
        data = tensor.detach()
        if data.numel() == 0:
            logger.info(
                "[SANA-WM diagnostics] %s: shape=%s dtype=%s device=%s empty",
                label,
                tuple(data.shape),
                data.dtype,
                data.device,
            )
            return

        numel = data.numel()
        sampled = numel > _SANA_WM_DIAGNOSTIC_MAX_EXACT_ELEMENTS
        sample_stride = 1
        if sampled:
            sample_stride = max(
                1, math.ceil(numel / _SANA_WM_DIAGNOSTIC_MAX_SAMPLE_ELEMENTS)
            )
            indices = torch.arange(0, numel, sample_stride, device=data.device)
            stats = torch.take(data, indices).float()
        else:
            stats = data.float().reshape(-1)
        finite = torch.isfinite(stats)
        finite_ratio = float(finite.float().mean().item())
        finite_stats = stats[finite] if bool(finite.any().item()) else stats
        flat = finite_stats.reshape(-1)
        stride = max(1, flat.numel() // 4096)
        fingerprint = float(flat[::stride].sum().item())
        std = float(finite_stats.std(unbiased=False).item())
        logger.info(
            "[SANA-WM diagnostics] %s: shape=%s dtype=%s device=%s "
            "finite=%.6f min=%.6g max=%.6g mean=%.6g std=%.6g "
            "l2=%.6g fingerprint=%.6g sampled=%s sample_stride=%d sample_size=%d",
            label,
            tuple(data.shape),
            data.dtype,
            data.device,
            finite_ratio,
            float(finite_stats.min().item()),
            float(finite_stats.max().item()),
            float(finite_stats.mean().item()),
            std,
            float(torch.linalg.vector_norm(finite_stats).item()),
            fingerprint,
            sampled,
            sample_stride,
            stats.numel(),
        )


def _resolve_sana_wm_vae_frame_tile_value(
    pipeline_config: Any,
    direct_attr: str,
    vae_config_attr: str,
    default: int,
) -> int:
    direct_value = getattr(pipeline_config, direct_attr, default)
    if direct_value != default:
        return int(direct_value)

    vae_config = getattr(pipeline_config, "vae_config", None)
    nested_value = getattr(vae_config, vae_config_attr, None)
    return int(nested_value or default)


def configure_sana_wm_ltx2_vae_for_long_video(
    vae: Any,
    pipeline_config: Any,
    *,
    log_info: Any | None = None,
) -> None:
    min_frames = _resolve_sana_wm_vae_frame_tile_value(
        pipeline_config,
        "vae_tile_sample_min_num_frames",
        "tile_sample_min_num_frames",
        _SANA_WM_DEFAULT_VAE_TILE_MIN_FRAMES,
    )
    stride_frames = _resolve_sana_wm_vae_frame_tile_value(
        pipeline_config,
        "vae_tile_sample_stride_num_frames",
        "tile_sample_stride_num_frames",
        _SANA_WM_DEFAULT_VAE_TILE_STRIDE_FRAMES,
    )

    use_tiling = bool(getattr(pipeline_config, "vae_tiling", True))
    if use_tiling and hasattr(vae, "enable_tiling"):
        try:
            vae.enable_tiling(
                tile_sample_min_num_frames=min_frames,
                tile_sample_stride_num_frames=stride_frames,
            )
        except TypeError:
            vae.enable_tiling()

    if hasattr(vae, "use_framewise_encoding"):
        vae.use_framewise_encoding = bool(
            getattr(pipeline_config, "vae_framewise_encoding", True)
        )
    if hasattr(vae, "use_framewise_decoding"):
        vae.use_framewise_decoding = bool(
            getattr(pipeline_config, "vae_framewise_decoding", True)
        )

    if hasattr(vae, "tile_sample_min_num_frames"):
        vae.tile_sample_min_num_frames = min_frames
    if hasattr(vae, "tile_sample_stride_num_frames"):
        vae.tile_sample_stride_num_frames = stride_frames

    if log_info is not None:
        log_info(
            "SANA-WM VAE tiling configured: spatial=%s, framewise_encode=%s, "
            "framewise_decode=%s, tile_frames_min=%d, tile_frames_stride=%d",
            getattr(vae, "use_tiling", use_tiling),
            getattr(vae, "use_framewise_encoding", None),
            getattr(vae, "use_framewise_decoding", None),
            getattr(vae, "tile_sample_min_num_frames", min_frames),
            getattr(vae, "tile_sample_stride_num_frames", stride_frames),
        )


def _sana_wm_effective_guidance_scale(batch: Any) -> float:
    cfg_scale = getattr(batch, "true_cfg_scale", None)
    if cfg_scale is None:
        cfg_scale = getattr(batch, "guidance_scale", 1.0)
    if cfg_scale is None:
        return 1.0
    return float(cfg_scale)


def _sana_wm_has_negative_condition(batch: Any) -> bool:
    """Return True when a negative/unconditional CFG condition is present."""
    neg_prompt = getattr(batch, "negative_prompt", None)
    if neg_prompt is not None:
        if isinstance(neg_prompt, (list, tuple)):
            return len(neg_prompt) > 0
        return True

    neg_embeds = getattr(batch, "negative_prompt_embeds", None)
    if neg_embeds is None:
        return False
    if isinstance(neg_embeds, torch.Tensor):
        return True
    try:
        return len(neg_embeds) > 0
    except TypeError:
        return bool(neg_embeds)


def _sana_wm_should_do_cfg(batch: Any) -> bool:
    return bool(getattr(batch, "do_classifier_free_guidance", False)) or (
        _sana_wm_effective_guidance_scale(batch) > 1.0
        and _sana_wm_has_negative_condition(batch)
    )


def sana_wm_stage_tp_world_size() -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    try:
        return get_tp_world_size()
    except AssertionError:
        return 1


def sana_wm_stage_tp_rank() -> int:
    if sana_wm_stage_tp_world_size() <= 1:
        return 0
    try:
        return get_tp_rank()
    except AssertionError:
        return 0


def sana_wm_broadcast_tensor_dict_from_tp_rank0(
    tensor_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    if sana_wm_stage_tp_world_size() <= 1:
        if tensor_dict is None:
            raise RuntimeError("SANA-WM TP broadcast payload is missing on rank 0.")
        return tensor_dict
    broadcasted = get_tp_group().broadcast_tensor_dict(tensor_dict, src=0)
    if broadcasted is None:
        raise RuntimeError("SANA-WM TP broadcast returned no payload.")
    return broadcasted


def _pack_sana_wm_text_outputs(
    outputs: tuple[
        list[torch.Tensor],
        list[torch.Tensor | None],
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[int]],
    ],
) -> dict[str, Any]:
    embeds, masks, pooled, embeds_masks, seq_lens = outputs
    return {
        "embeds_count": len(embeds),
        "embeds": {str(index): tensor for index, tensor in enumerate(embeds)},
        "masks_count": len(masks),
        "masks": {str(index): tensor for index, tensor in enumerate(masks)},
        "pooled_count": len(pooled),
        "pooled": {str(index): tensor for index, tensor in enumerate(pooled)},
        "embeds_masks_count": len(embeds_masks),
        "embeds_masks": {
            str(index): tensor for index, tensor in enumerate(embeds_masks)
        },
        "seq_lens": seq_lens,
    }


def _unpack_sana_wm_text_outputs(
    payload: dict[str, Any],
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor | None],
    list[torch.Tensor],
    list[torch.Tensor],
    list[list[int]],
]:
    def ordered_tensors(name: str) -> list[Any]:
        values = payload.get(name, {})
        return [
            values[str(index)] for index in range(int(payload.get(f"{name}_count", 0)))
        ]

    return (
        ordered_tensors("embeds"),
        ordered_tensors("masks"),
        ordered_tensors("pooled"),
        ordered_tensors("embeds_masks"),
        payload.get("seq_lens", []),
    )


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value if isinstance(value, torch.Tensor) else None


def _to_device_dtype(
    value: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if value is None:
        return None
    if dtype is None:
        return value.to(device=device)
    return value.to(device=device, dtype=dtype)


def _cat_optional_tensors(
    neg: torch.Tensor | None,
    pos: torch.Tensor | None,
) -> torch.Tensor | None:
    if neg is None and pos is None:
        return None
    if neg is None:
        return pos
    if pos is None:
        return neg
    return torch.cat([neg, pos], dim=0)


def _text_sequence_dim(tensor: torch.Tensor) -> int:
    return -2 if tensor.ndim >= 3 else -1


def _pad_text_sequence(
    tensor: torch.Tensor | None,
    target_length: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    seq_dim = _text_sequence_dim(tensor)
    current_length = tensor.shape[seq_dim]
    if current_length == target_length:
        return tensor
    if current_length > target_length:
        index = [slice(None)] * tensor.ndim
        index[seq_dim] = slice(0, target_length)
        return tensor[tuple(index)]

    pad_shape = list(tensor.shape)
    pad_shape[seq_dim] = target_length - current_length
    padding = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=seq_dim)


def _align_sana_wm_cfg_text_conditions(
    pos_embeds: torch.Tensor,
    neg_embeds: torch.Tensor | None,
    pos_mask: torch.Tensor | None,
    neg_mask: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if neg_embeds is None:
        return pos_embeds, neg_embeds, pos_mask, neg_mask

    target_length = max(pos_embeds.shape[-2], neg_embeds.shape[-2])
    pos_embeds = _pad_text_sequence(pos_embeds, target_length)
    neg_embeds = _pad_text_sequence(neg_embeds, target_length)

    if pos_mask is None:
        pos_mask = torch.ones(
            (pos_embeds.shape[0], pos_embeds.shape[-2]),
            device=pos_embeds.device,
            dtype=torch.long,
        )
    if neg_mask is None:
        neg_mask = torch.ones(
            (neg_embeds.shape[0], neg_embeds.shape[-2]),
            device=neg_embeds.device,
            dtype=torch.long,
        )
    pos_mask = _pad_text_sequence(pos_mask, target_length)
    neg_mask = _pad_text_sequence(neg_mask, target_length)
    return pos_embeds, neg_embeds, pos_mask, neg_mask


def sana_wm_tensor_or_tensor_list(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return V.is_tensor(value)
    return V.list_of_tensors(value)


def sana_wm_optional_tensor_or_tensor_list_allow_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return V.is_tensor(value)
    if isinstance(value, (list, tuple)):
        return all(
            isinstance(item, torch.Tensor) and V.is_tensor(item) for item in value
        )
    return False


_sana_wm_tensor_or_tensor_list = sana_wm_tensor_or_tensor_list
_sana_wm_optional_tensor_or_tensor_list_allow_empty = (
    sana_wm_optional_tensor_or_tensor_list_allow_empty
)


_SANA_WM_TORCH_COMPILE_DEFAULT_CACHE_SIZE_LIMIT = 128
_SANA_WM_TORCH_COMPILE_DEFAULT_SCOPE = "regional"
_SANA_WM_TORCH_COMPILE_SCOPES = ("regional", "full", "off")


def _normalize_sana_wm_torch_compile_scope(value: Any) -> str:
    if value is not None:
        value = str(value).strip().lower().replace("-", "_")
        aliases = {
            "0": "off",
            "false": "off",
            "no": "off",
            "none": "off",
            "block": "regional",
            "blocks": "regional",
            "regional_blocks": "regional",
            "module": "full",
            "transformer": "full",
            "full_module": "full",
        }
        value = aliases.get(value, value)
    mode = _SANA_WM_TORCH_COMPILE_DEFAULT_SCOPE if value is None else str(value)
    if mode in _SANA_WM_TORCH_COMPILE_SCOPES:
        return mode
    logger.warning(
        "Ignoring invalid sana_wm_torch_compile_scope=%r. Expected one of %s; "
        "using %r.",
        value,
        sorted(_SANA_WM_TORCH_COMPILE_SCOPES),
        _SANA_WM_TORCH_COMPILE_DEFAULT_SCOPE,
    )
    return _SANA_WM_TORCH_COMPILE_DEFAULT_SCOPE


def _sana_wm_nonempty_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


class SanaWMDecodingStage(DecodingStage):
    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify SANA-WM decoding stage inputs."""
        result = super().verify_input(batch, server_args)
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
        *,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor:
        configure_sana_wm_ltx2_vae_for_long_video(
            self.vae,
            server_args.pipeline_config,
            log_info=self.log_info,
        )
        frames = super().decode(latents, server_args, vae_dtype=vae_dtype)
        log_sana_wm_tensor_stats("decode.frames", frames, server_args.pipeline_config)
        return frames


def _sana_wm_camera_conditions_ready(value: Any) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, torch.Tensor)
        and V.is_tensor(value)
        and value.dim() == 3
        and value.shape[-1] == 20
    )


def _sana_wm_chunk_plucker_ready(value: Any) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, torch.Tensor)
        and V.is_tensor(value)
        and value.dim() == 5
        and value.shape[1] == 48
    )


class SanaWMTextEncodingStage(TextEncodingStage):
    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        if sana_wm_stage_tp_world_size() > 1 and sana_wm_stage_tp_rank() != 0:
            return []
        return super().component_uses(server_args, stage_name)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify SANA-WM text encoding stage inputs."""
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        result.add_check("text_encoders", self.text_encoders, lambda x: len(x) == 1)
        result.add_check("tokenizers", self.tokenizers, lambda x: len(x) == 1)
        result.add_check(
            "do_classifier_free_guidance",
            batch.do_classifier_free_guidance,
            V.bool_value,
        )
        result.add_check("prompt_embeds", batch.prompt_embeds, V.is_list)
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            _sana_wm_optional_tensor_or_tensor_list_allow_empty,
        )
        result.add_check(
            "negative_prompt",
            batch.negative_prompt,
            lambda x: (
                not batch.do_classifier_free_guidance
                or _first_tensor(batch.negative_prompt_embeds) is not None
                or V.string_or_list_strings(x)
            ),
        )
        return result

    @staticmethod
    def _text_encoder_max_length(server_args: ServerArgs) -> int:
        encoder_cfg = server_args.pipeline_config.text_encoder_configs[0]
        arch_config = getattr(encoder_cfg, "arch_config", None)
        return int(getattr(arch_config, "text_len", 300) or 300)

    @staticmethod
    def _chi_prompt(server_args: ServerArgs) -> str:
        parts = getattr(server_args.pipeline_config, "chi_prompt", ()) or ()
        return "\n".join(parts)

    @staticmethod
    def _select_prompt_window(
        tensor: torch.Tensor | None,
        max_length: int,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        seq_dim = _text_sequence_dim(tensor)
        if tensor.shape[seq_dim] <= max_length:
            return tensor
        index = [slice(None)] * tensor.ndim
        tail_start = tensor.shape[seq_dim] - max_length + 1
        select = torch.cat(
            [
                torch.zeros(1, device=tensor.device, dtype=torch.long),
                torch.arange(
                    tail_start,
                    tensor.shape[seq_dim],
                    device=tensor.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        index[seq_dim] = select
        return tensor[tuple(index)]

    @staticmethod
    def _seq_lens_from_masks(masks: list[torch.Tensor | None]) -> list[list[int]]:
        seq_lens = []
        for mask in masks:
            if mask is None:
                seq_lens.append([])
            else:
                seq_lens.append([int(x) for x in mask.long().sum(dim=-1).tolist()])
        return seq_lens

    def _encode_text_on_tp_rank0(
        self,
        *args,
        **kwargs,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor | None],
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[int]],
    ]:
        if sana_wm_stage_tp_world_size() <= 1:
            return self.encode_text(*args, **kwargs)

        payload = None
        if sana_wm_stage_tp_rank() == 0:
            payload = _pack_sana_wm_text_outputs(self.encode_text(*args, **kwargs))
        payload = sana_wm_broadcast_tensor_dict_from_tp_rank0(payload)
        return _unpack_sana_wm_text_outputs(payload)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if len(self.text_encoders) != 1:
            raise ValueError(
                "SANA-WM stage-1 expects exactly one Gemma-2 text encoder."
            )
        assert batch.prompt is not None

        max_length = self._text_encoder_max_length(server_args)
        chi_prompt = self._chi_prompt(server_args)
        prompt_text = batch.prompt
        if isinstance(prompt_text, str):
            prompt_text = [prompt_text]
        else:
            prompt_text = list(prompt_text)

        tokenizer = self.tokenizers[0]
        if chi_prompt:
            prompt_text = [chi_prompt + text for text in prompt_text]
            max_length_all = len(tokenizer.encode(chi_prompt)) + max_length - 2
        else:
            max_length_all = max_length

        (
            prompt_embeds_list,
            prompt_masks_list,
            pooler_embeds_list,
            prompt_embeds_masks_list,
            _prompt_seq_lens_list,
        ) = self._encode_text_on_tp_rank0(
            prompt_text,
            server_args,
            encoder_index=[0],
            return_attention_mask=True,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
        )

        prompt_embeds_list = [
            self._select_prompt_window(tensor, max_length)
            for tensor in prompt_embeds_list
        ]
        prompt_masks_list = [
            self._select_prompt_window(tensor, max_length)
            for tensor in prompt_masks_list
        ]
        prompt_embeds_masks_list = [
            self._select_prompt_window(tensor, max_length)
            for tensor in prompt_embeds_masks_list
        ]
        prompt_seq_lens_list = self._seq_lens_from_masks(prompt_masks_list)

        has_preencoded_negative = (
            _first_tensor(getattr(batch, "negative_prompt_embeds", None)) is not None
        )
        # Pre-initialise so references below are always bound regardless of which branch runs.
        neg_embeds_list: list[torch.Tensor] = []
        neg_masks_list: list[torch.Tensor] = []
        neg_pooler_embeds_list: list[torch.Tensor] = []
        neg_embeds_masks_list: list[torch.Tensor] = []
        neg_seq_lens_list: list[torch.Tensor] = []

        if batch.do_classifier_free_guidance and not has_preencoded_negative:
            negative_prompt = batch.negative_prompt
            if not isinstance(negative_prompt, (str, list)) or (
                isinstance(negative_prompt, list)
                and not all(isinstance(text, str) for text in negative_prompt)
            ):
                raise TypeError(
                    "SANA-WM CFG negative_prompt must be a string or a list of "
                    f"strings, got {type(negative_prompt).__name__}."
                )
            (
                neg_embeds_list,
                neg_masks_list,
                neg_pooler_embeds_list,
                neg_embeds_masks_list,
                _neg_seq_lens_list,
            ) = self._encode_text_on_tp_rank0(
                negative_prompt,
                server_args,
                encoder_index=[0],
                return_attention_mask=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            neg_seq_lens_list = self._seq_lens_from_masks(neg_masks_list)

        self._append_positive_text_outputs(
            batch,
            prompt_embeds_list,
            prompt_masks_list,
            pooler_embeds_list,
            prompt_embeds_masks_list,
            prompt_seq_lens_list,
        )

        if batch.do_classifier_free_guidance:
            if has_preencoded_negative:
                self._align_preencoded_negative_text_outputs(batch, prompt_embeds_list)
            else:
                self._append_negative_text_outputs(
                    batch,
                    prompt_embeds_list,
                    neg_embeds_list,
                    neg_masks_list,
                    neg_pooler_embeds_list,
                    neg_embeds_masks_list,
                    neg_seq_lens_list,
                )

        self.log_info(
            "SANA-WM text encoded with chi_prompt=%s, prompt_window=%d, "
            "positive_raw_window=%d",
            "yes" if chi_prompt else "no",
            max_length,
            max_length_all,
        )
        log_sana_wm_tensor_stats(
            "text.prompt_embeds",
            prompt_embeds_list[0],
            server_args.pipeline_config,
        )
        if batch.do_classifier_free_guidance:
            neg_for_log = (
                _first_tensor(batch.negative_prompt_embeds)
                if has_preencoded_negative
                else neg_embeds_list[0]
            )
            log_sana_wm_tensor_stats(
                "text.negative_prompt_embeds",
                neg_for_log,
                server_args.pipeline_config,
            )

        return batch


class SanaWMDenoisingStage(DenoisingStage):
    @property
    def parallelism_type(self) -> StageParallelismType:
        if self.server_args.enable_cfg_parallel:
            return StageParallelismType.CFG_PARALLEL
        return StageParallelismType.REPLICATED

    def _maybe_enable_torch_compile(self, module: object) -> None:
        """Regionally compile SANA-WM blocks while keeping GDN/UCPE attention eager."""
        if not self.server_args.enable_torch_compile or not isinstance(
            module, nn.Module
        ):
            return
        if envs.SGLANG_CACHE_DIT_ENABLED and not self._cache_dit_enabled:
            logger.debug(
                "Deferring SANA-WM regional torch.compile until cache-dit is enabled"
            )
            return
        module_id = id(module)
        if module_id in self._torch_compiled_module_ids:
            return

        pipeline_config = getattr(self.server_args, "pipeline_config", None)
        compile_scope = _normalize_sana_wm_torch_compile_scope(
            getattr(
                pipeline_config,
                "sana_wm_torch_compile_scope",
                _SANA_WM_TORCH_COMPILE_DEFAULT_SCOPE,
            )
        )
        if compile_scope == "off":
            logger.info(
                "SANA-WM torch.compile disabled by "
                "pipeline_config.sana_wm_torch_compile_scope=off."
            )
            return
        if compile_scope == "full":
            logger.warning(
                "SANA-WM full-module torch.compile requested by "
                "pipeline_config.sana_wm_torch_compile_scope=full. "
                "The default regional scope is safer because GDN/UCPE attention "
                "contains Triton launch-shape logic that should remain eager."
            )
            super()._maybe_enable_torch_compile(module)
            return

        repeated_block_names = tuple(getattr(module, "_repeated_blocks", ()) or ())
        if not repeated_block_names:
            super()._maybe_enable_torch_compile(module)
            return

        compile_targets = [
            submodule
            for submodule in module.modules()
            if submodule.__class__.__name__ in repeated_block_names
        ]
        if not compile_targets:
            logger.warning(
                "SANA-WM regional torch.compile skipped: repeated block classes "
                "%s were not found in %s.",
                repeated_block_names,
                module.__class__.__name__,
            )
            return

        compile_kwargs: dict[str, Any] = {"fullgraph": False, "dynamic": None}
        if current_platform.is_npu():
            compile_kwargs["backend"] = get_compiler_backend()
            compile_kwargs["dynamic"] = False
            logger.info(
                "Regionally compiling SANA-WM blocks with torchair backend on NPU"
            )
        else:
            self._maybe_raise_torch_compile_cache_limit(
                len(compile_targets),
                pipeline_config,
            )
            try:
                import torch._inductor.config as _inductor_cfg

                _inductor_cfg.reorder_for_compute_comm_overlap = True
            except ImportError:
                pass
            # Match vLLM's regional compile default: torch.compile(dynamic=True)
            # with no explicit mode. Users can still opt into max-autotune or
            # another mode with the SANA-WM pipeline config, or the global one.
            mode = getattr(
                pipeline_config,
                "sana_wm_torch_compile_mode",
                None,
            ) or _sana_wm_nonempty_env("SGLANG_TORCH_COMPILE_MODE")
            if mode is not None:
                compile_kwargs["mode"] = mode
            compile_kwargs["dynamic"] = True
            logger.info(
                "Regionally compiling %d SANA-WM blocks with mode=%s",
                len(compile_targets),
                mode or "default",
            )

        if self._needs_nvfp4_jit_prewarm(module):
            logger.info(
                "Prewarming NVFP4 JIT modules before SANA-WM regional torch.compile "
                "to avoid Dynamo tracing JIT initialization."
            )
            prewarm_nvfp4_jit_modules()

        for submodule in compile_targets:
            submodule.compile(**compile_kwargs)

        logger.info(
            "Regionally compiled %d SANA-WM blocks with torch.compile kwargs: %s",
            len(compile_targets),
            compile_kwargs,
        )
        self._torch_compiled_module_ids.add(module_id)

    @staticmethod
    def _maybe_raise_torch_compile_cache_limit(
        num_compile_targets: int,
        pipeline_config: Any | None = None,
    ) -> None:
        try:
            import torch._dynamo.config as _dynamo_cfg
        except ImportError:
            return

        configured_limit = int(
            getattr(
                pipeline_config,
                "sana_wm_torch_compile_cache_size_limit",
                _SANA_WM_TORCH_COMPILE_DEFAULT_CACHE_SIZE_LIMIT,
            )
        )

        target_limit = max(configured_limit, num_compile_targets * 4)
        current_limit = getattr(_dynamo_cfg, "cache_size_limit", 64)
        if current_limit < target_limit:
            _dynamo_cfg.cache_size_limit = target_limit

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Run the SANA-WM denoising loop.

        Wraps the base-class ``forward`` so request-local model caches are freed
        even when a denoising step raises an exception.
        """
        try:
            return super().forward(batch, server_args)
        except BaseException:
            _clear_sana_wm_request_runtime_cache(batch)
            raise

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify SANA-WM denoising stage inputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])
        result.add_check(
            "prompt_embeds",
            batch.prompt_embeds,
            _sana_wm_tensor_or_tensor_list,
        )
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check(
            "guidance_scale",
            _sana_wm_effective_guidance_scale(batch),
            V.non_negative_float,
        )
        result.add_check("eta", batch.eta, V.non_negative_float)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check(
            "do_classifier_free_guidance",
            batch.do_classifier_free_guidance,
            V.bool_value,
        )
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            (
                _sana_wm_tensor_or_tensor_list
                if _sana_wm_should_do_cfg(batch)
                else _sana_wm_optional_tensor_or_tensor_list_allow_empty
            ),
        )
        extra = getattr(batch, "extra", None)
        result.add_check("extra", extra, lambda x: x is None or isinstance(x, dict))
        extra = extra or {}
        result.add_check(
            "camera_conditions",
            extra.get("camera_conditions"),
            _sana_wm_camera_conditions_ready,
        )
        result.add_check(
            "chunk_plucker",
            extra.get("chunk_plucker"),
            _sana_wm_chunk_plucker_ready,
        )
        return result

    @staticmethod
    def _write_serial_cfg_latent_model_input(
        buffer: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = latents.shape[0]
        expected_shape = (batch_size * 2, *latents.shape[1:])
        if tuple(buffer.shape) != expected_shape:
            raise ValueError(
                "SANA-WM serial CFG latent buffer shape mismatch: expected "
                f"{expected_shape}, got {tuple(buffer.shape)}."
            )
        buffer[:batch_size].copy_(latents)
        buffer[batch_size:].copy_(latents)
        return buffer

    @staticmethod
    def _prepare_step_timesteps(
        step_timestep: torch.Tensor,
        frame_condition_limit: torch.Tensor,
        token_condition_limit: torch.Tensor,
        *,
        do_cfg: bool,
        cfg_parallel: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = step_timestep.float()
        model_timestep = torch.minimum(
            timestep.expand(frame_condition_limit.shape),
            frame_condition_limit,
        )
        if do_cfg and not cfg_parallel:
            model_timestep = torch.cat([model_timestep, model_timestep], dim=0)
        per_token_timesteps = torch.minimum(
            timestep.expand(token_condition_limit.shape),
            token_condition_limit,
        )
        return model_timestep, per_token_timesteps

    @staticmethod
    def _combine_serial_cfg_noise_in_place(
        noise_pred: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        return (
            noise_pred_text.sub_(noise_pred_uncond)
            .mul_(guidance_scale)
            .add_(noise_pred_uncond)
        )

    @staticmethod
    def _combine_cfg_parallel_noise(
        noise_pred: torch.Tensor,
        guidance_scale: float,
        cfg_rank: int,
    ) -> torch.Tensor:
        """Combine CFG branches across exactly 2 CFG-parallel ranks.

        Rank 0 holds the positive-branch prediction, rank 1 holds the
        negative-branch prediction.  The all-reduce sums the two scaled
        contributions so every rank ends up with the full CFG output.

        ``cfg_world_size != 2`` is rejected at the start of the denoising loop,
        so ``cfg_rank`` is always 0 or 1 here.
        """
        if cfg_rank not in (0, 1):
            raise ValueError(
                "SANA-WM CFG parallel combine expects cfg_rank 0 or 1, "
                f"got cfg_rank={cfg_rank}."
            )
        if cfg_rank == 0:
            partial = guidance_scale * noise_pred
        else:
            partial = (1.0 - guidance_scale) * noise_pred
        return cfg_model_parallel_all_reduce(partial)

    def _prepare_denoising_loop(
        self, batch: Req, server_args: ServerArgs
    ) -> DenoisingContext:
        if batch.latents is None:
            raise ValueError("SANA-WM denoising requires initialized latents.")
        if batch.latents.ndim != 5:
            raise ValueError(
                "SANA-WM denoising expects 5D latents shaped (B, C, T, H, W), "
                f"got {tuple(batch.latents.shape)}."
            )

        device = get_local_torch_device()
        target_dtype = PRECISION_TO_TYPE.get(
            getattr(server_args.pipeline_config, "dit_precision", "bf16"),
            torch.bfloat16,
        )
        scheduler = getattr(
            batch, "scheduler", None
        ) or get_or_create_request_scheduler(batch, self.scheduler)
        timesteps = batch.timesteps
        if timesteps is None:
            raise ValueError("SANA-WM denoising requires prepared timesteps.")

        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        _clear_sana_wm_request_runtime_cache(batch)
        self._maybe_enable_cache_dit_and_torch_compile(num_inference_steps, batch)

        latents = batch.latents.to(device=device, dtype=target_dtype)
        init_condition_latents = latents[:, :, :1].clone()
        condition_mask = torch.zeros(
            (
                latents.shape[0],
                1,
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )
        condition_mask[:, :, :1] = 1

        pos_embeds = _to_device_dtype(
            _first_tensor(server_args.pipeline_config.get_pos_prompt_embeds(batch)),
            device=device,
            dtype=target_dtype,
        )
        pos_mask = _to_device_dtype(
            _first_tensor(batch.prompt_attention_mask), device=device
        )
        if pos_embeds is None:
            raise ValueError("SANA-WM denoising requires positive prompt embeds.")

        do_cfg = _sana_wm_should_do_cfg(batch)
        guidance_scale = _sana_wm_effective_guidance_scale(batch)
        neg_embeds = None
        neg_mask = None
        if do_cfg:
            neg_embeds = _to_device_dtype(
                _first_tensor(server_args.pipeline_config.get_neg_prompt_embeds(batch)),
                device=device,
                dtype=target_dtype,
            )
            neg_mask = _to_device_dtype(
                _first_tensor(batch.negative_attention_mask), device=device
            )
            if neg_embeds is None:
                raise ValueError("SANA-WM CFG requires negative prompt embeds.")

            pos_embeds, neg_embeds, pos_mask, neg_mask = (
                _align_sana_wm_cfg_text_conditions(
                    pos_embeds, neg_embeds, pos_mask, neg_mask
                )
            )

        extra = batch.extra or {}
        chunk_kwargs = {}
        for key in ("chunk_index", "chunk_size", "chunk_split_strategy"):
            value = extra.get(key)
            if value is not None:
                chunk_kwargs[key] = value
        camera_conditions = _to_device_dtype(
            extra.get("camera_conditions"), device=device, dtype=target_dtype
        )
        chunk_plucker = _to_device_dtype(
            extra.get("chunk_plucker"), device=device, dtype=target_dtype
        )

        cfg_parallel = bool(server_args.enable_cfg_parallel and do_cfg)
        cfg_rank = get_classifier_free_guidance_rank() if cfg_parallel else 0
        if cfg_parallel:
            cfg_world_size = get_classifier_free_guidance_world_size()
            if cfg_world_size != 2:
                raise ValueError(
                    f"SANA-WM CFG parallel requires exactly 2 CFG ranks (one for "
                    f"the positive branch, one for the negative branch), but "
                    f"cfg_world_size={cfg_world_size}. "
                    "Set --cfg-parallel-size 2 (or disable with --no-cfg-parallel)."
                )

        if cfg_parallel:
            if cfg_rank == 1:
                branch_embeds = neg_embeds
                branch_mask = neg_mask
            else:
                branch_embeds = pos_embeds
                branch_mask = pos_mask
            model_kwargs = {
                "encoder_hidden_states": branch_embeds,
                "encoder_attention_mask": branch_mask,
                "camera_conditions": camera_conditions,
                "chunk_plucker": chunk_plucker,
            }
        else:
            # Serial CFG doubles latent/text batches during denoising, but camera
            # and Plucker tensors are request-static. The transformer camera ops
            # accept a smaller static batch and repeat it against the 2B latent
            # batch, avoiding an extra materialized copy here.
            model_kwargs = {
                "encoder_hidden_states": (
                    torch.cat([neg_embeds, pos_embeds], dim=0) if do_cfg else pos_embeds
                ),
                "encoder_attention_mask": (
                    _cat_optional_tensors(neg_mask, pos_mask) if do_cfg else pos_mask
                ),
                "camera_conditions": camera_conditions,
                "chunk_plucker": chunk_plucker,
            }
        model_kwargs.update(chunk_kwargs)
        model_kwargs = self.prepare_extra_func_kwargs(
            getattr(self.transformer, "forward", self.transformer),
            model_kwargs,
        )

        serial_cfg_latent_model_input = (
            torch.empty(
                (latents.shape[0] * 2, *latents.shape[1:]),
                device=latents.device,
                dtype=latents.dtype,
            )
            if do_cfg and not cfg_parallel
            else None
        )
        timestep_frame_condition_limit = (
            1.0 - condition_mask[:, :, :, 0, 0].float()
        ) * 1000.0
        timestep_token_condition_limit = (
            1.0 - condition_mask.flatten(2).squeeze(1).float()
        ) * 1000.0

        self.log_info(
            "SANA-WM flow_euler_ltx denoising: latent=%s, steps=%d, cfg=%s, "
            "cfg_parallel=%s, guidance_scale=%.4f, first_frame_locked=yes",
            tuple(latents.shape),
            len(timesteps),
            do_cfg,
            cfg_parallel,
            guidance_scale,
        )
        log_sana_wm_tensor_stats(
            "denoise.input_latents",
            latents,
            server_args.pipeline_config,
        )

        return DenoisingContext(
            scheduler=scheduler,
            extra_step_kwargs={},
            target_dtype=target_dtype,
            autocast_enabled=(
                bool(getattr(server_args.pipeline_config, "enable_autocast", False))
                and target_dtype != torch.float32
                and not getattr(server_args, "disable_autocast", False)
            ),
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
            image_kwargs={},
            pos_cond_kwargs={},
            neg_cond_kwargs={},
            latents=latents,
            boundary_timestep=None,
            z=None,
            reserved_frames_mask=None,
            seq_len=None,
            guidance=torch.empty(0, device=device, dtype=target_dtype),
            is_warmup=batch.is_warmup,
            cfg_policy=None,
            extra={
                "cfg_parallel": cfg_parallel,
                "cfg_rank": cfg_rank,
                "condition_mask": condition_mask,
                "do_cfg": do_cfg,
                "guidance_scale": guidance_scale,
                "init_condition_latents": init_condition_latents,
                "model_kwargs": model_kwargs,
                "serial_cfg_latent_model_input": serial_cfg_latent_model_input,
                "start_time": time.perf_counter(),
                "timestep_frame_condition_limit": timestep_frame_condition_limit,
                "timestep_token_condition_limit": timestep_token_condition_limit,
            },
        )

    def _prepare_step_attn_metadata(
        self,
        ctx: DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        step_index: int,
        t_int: int,
        timesteps_cpu: torch.Tensor,
    ) -> Any | None:
        return None

    def _run_denoising_step(
        self,
        ctx: DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        cfg_parallel = bool(ctx.extra["cfg_parallel"])
        cfg_rank = int(ctx.extra["cfg_rank"])
        condition_mask = ctx.extra["condition_mask"]
        do_cfg = bool(ctx.extra["do_cfg"])
        model_kwargs = ctx.extra["model_kwargs"]

        if cfg_parallel:
            latent_model_input = ctx.latents
        elif do_cfg:
            latent_model_input = self._write_serial_cfg_latent_model_input(
                ctx.extra["serial_cfg_latent_model_input"],
                ctx.latents,
            )
        else:
            latent_model_input = ctx.latents

        model_timestep, per_token_timesteps = self._prepare_step_timesteps(
            step.t_device,
            ctx.extra["timestep_frame_condition_limit"],
            ctx.extra["timestep_token_condition_limit"],
            do_cfg=do_cfg,
            cfg_parallel=cfg_parallel,
        )

        with set_forward_context(
            current_timestep=step.step_index,
            attn_metadata=None,
            forward_batch=batch,
        ):
            noise_pred = step.current_model(
                hidden_states=latent_model_input.to(ctx.target_dtype),
                timestep=model_timestep,
                **model_kwargs,
            )

        if do_cfg:
            guidance_scale = float(ctx.extra["guidance_scale"])
            if cfg_parallel:
                noise_pred = self._combine_cfg_parallel_noise(
                    noise_pred, guidance_scale, cfg_rank
                )
            else:
                noise_pred = self._combine_serial_cfg_noise_in_place(
                    noise_pred,
                    guidance_scale,
                )

        latents_dtype = ctx.latents.dtype
        latents_shape = ctx.latents.shape
        batch_size, channels, _, _, _ = latents_shape
        scheduler_output = ctx.scheduler.step(
            -noise_pred.reshape(batch_size, channels, -1).transpose(1, 2),
            step.t_device,
            ctx.latents.reshape(batch_size, channels, -1).transpose(1, 2),
            per_token_timesteps=per_token_timesteps,
            return_dict=False,
        )[0]
        denoised_latents = scheduler_output.transpose(1, 2).reshape(latents_shape)

        tokens_to_denoise = step.t_device.float() / 1000.0 - 1e-6 < (
            1.0 - condition_mask
        )
        ctx.latents = torch.where(tokens_to_denoise, denoised_latents, ctx.latents)
        if ctx.latents.dtype != latents_dtype:
            ctx.latents = ctx.latents.to(latents_dtype)

        if sana_wm_diagnostics_enabled(server_args.pipeline_config) and (
            step.step_index == 0 or step.step_index == len(ctx.timesteps) - 1
        ):
            log_sana_wm_tensor_stats(
                f"denoise.step_{step.step_index}.noise_pred",
                noise_pred,
                server_args.pipeline_config,
            )
            log_sana_wm_tensor_stats(
                f"denoise.step_{step.step_index}.latents",
                ctx.latents,
                server_args.pipeline_config,
            )

    def _finalize_denoising_loop(
        self, ctx: DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        _clear_sana_wm_request_runtime_cache(batch)

        log_sana_wm_tensor_stats(
            "denoise.output_latents",
            ctx.latents,
            server_args.pipeline_config,
        )
        unchanged = (
            (ctx.latents[:, :, :1] - ctx.extra["init_condition_latents"])
            .abs()
            .max()
            .item()
        )
        self.log_info(
            "SANA-WM flow_euler_ltx denoising finished in %.4f seconds; "
            "first_frame_max_delta=%.6g",
            time.perf_counter() - ctx.extra["start_time"],
            float(unchanged),
        )
        super()._finalize_denoising_loop(ctx, batch, server_args)


class SanaWMBeforeDenoisingStage(PipelineStage):

    def __init__(
        self,
        vae,
        scheduler,
        pipeline_config: Any,
    ):
        super().__init__()
        self.vae = vae
        self.scheduler = scheduler
        self.pipeline_config = pipeline_config

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        uses: list[ComponentUse] = []
        pipeline_config = getattr(server_args, "pipeline_config", self.pipeline_config)
        if self.vae is not None:
            vae_dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]
            uses.append(
                ComponentUse(
                    stage_name=stage_name,
                    component_name="vae",
                    target_dtype=vae_dtype,
                )
            )
        return uses

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify SANA-WM before-denoising stage inputs."""
        result = VerificationResult()
        pipeline_config = getattr(server_args, "pipeline_config", self.pipeline_config)
        vae_stride = getattr(pipeline_config, "vae_stride", (8, 32, 32))
        height_stride = int(vae_stride[1])
        width_stride = int(vae_stride[2] if len(vae_stride) > 2 else height_stride)

        result.add_check(
            "condition_image",
            getattr(batch, "condition_image", None),
            [V.not_none, lambda value: not isinstance(value, list) or len(value) > 0],
        )
        result.add_check(
            "height",
            batch.height,
            [V.positive_int, V.divisible(height_stride)],
        )
        result.add_check(
            "width",
            batch.width,
            [V.positive_int, V.divisible(width_stride)],
        )
        result.add_check(
            "num_frames",
            getattr(batch, "num_frames", None),
            lambda value: value is None or V.positive_int(value),
        )
        result.add_check(
            "num_inference_steps",
            batch.num_inference_steps,
            V.positive_int,
        )
        result.add_check(
            "prompt_embeds",
            batch.prompt_embeds,
            _sana_wm_tensor_or_tensor_list,
        )
        result.add_check(
            "guidance_scale",
            _sana_wm_effective_guidance_scale(batch),
            V.non_negative_float,
        )
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            (
                _sana_wm_tensor_or_tensor_list
                if _sana_wm_should_do_cfg(batch)
                else _sana_wm_optional_tensor_or_tensor_list_allow_empty
            ),
        )
        result.add_check(
            "condition_inputs",
            getattr(batch, "condition_inputs", None),
            sana_wm_condition_inputs_are_valid,
        )
        return result

    @torch.no_grad()
    def _vae_encode_image(
        self,
        image: torch.Tensor,  # (B, C, H, W) or (B, C, 1, H, W) in [0, 1] float
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode one or more first-frame images through the VAE encoder."""
        vae = self.vae
        configure_sana_wm_ltx2_vae_for_long_video(vae, self.pipeline_config)
        vae_dtype = PRECISION_TO_TYPE.get(
            self.pipeline_config.vae_precision, torch.bfloat16
        )

        # Normalize to [0, 1] then shift to [-1, 1] as expected by the VAE.
        # uint8 tensors are always in [0, 255]; for float tensors the caller
        # should supply [0, 1] values — we accept both for robustness.
        if not image.is_floating_point():
            image = image.float() / 255.0
        elif image.max() > 1.5:
            image = image / 255.0
        image = (image * 2.0 - 1.0).to(device=device, dtype=vae_dtype)

        # Add temporal dim if not present: (B, C, H, W) → (B, C, 1, H, W)
        if image.dim() == 4:
            image = image.unsqueeze(2)

        log_sana_wm_tensor_stats(
            "first_frame.pixel_input_normalized",
            image,
            self.pipeline_config,
        )
        with self.use_declared_component(
            component_name="vae",
            module=vae,
            target_dtype=vae_dtype,
        ) as active_vae:
            vae = active_vae if active_vae is not None else vae
            z = self._extract_vae_latents(vae.encode(image)).float()

        latents_mean = getattr(vae, "latents_mean", None)
        latents_std = getattr(vae, "latents_std", None)
        scaling_factor = self._get_vae_scaling_factor(vae)
        if sana_wm_diagnostics_enabled(self.pipeline_config):
            logger.info(
                "[SANA-WM diagnostics] VAE encode normalization: "
                "has_latents_mean_std=%s scaling_factor=%.6g",
                isinstance(latents_mean, torch.Tensor)
                and isinstance(latents_std, torch.Tensor),
                scaling_factor,
            )
        if isinstance(latents_mean, torch.Tensor) and isinstance(
            latents_std, torch.Tensor
        ):
            latents_mean = latents_mean.to(device=z.device, dtype=z.dtype).view(
                1, -1, 1, 1, 1
            )
            latents_std = latents_std.to(device=z.device, dtype=z.dtype).view(
                1, -1, 1, 1, 1
            )
            z = (z - latents_mean) * scaling_factor / latents_std
        else:
            # Legacy VAE convention: encode applies shift before scaling.
            shift_factor = getattr(vae, "shift_factor", None)
            if shift_factor is not None:
                z = z - (
                    shift_factor.to(z.device, z.dtype)
                    if isinstance(shift_factor, torch.Tensor)
                    else shift_factor
                )
            z = z * scaling_factor

        log_sana_wm_tensor_stats(
            "first_frame.latent_normalized",
            z,
            self.pipeline_config,
        )
        return z.to(dtype=dtype)  # (B, 128, 1, H_sp, W_sp)

    @staticmethod
    def _extract_vae_latents(encoded: Any) -> torch.Tensor:
        """Return deterministic VAE latents from common Diffusers encode() outputs.

        Handles DiagonalGaussianDistribution (latent_dist attribute), plain
        tensors, and tuple wrappers — up to 8 nesting levels to guard against
        infinite loops from unexpected encoder return types.
        """
        for _ in range(8):
            if isinstance(encoded, torch.Tensor):
                return encoded

            latent_dist = getattr(encoded, "latent_dist", None)
            if latent_dist is not None:
                if hasattr(latent_dist, "mode"):
                    return latent_dist.mode()
                mean = getattr(latent_dist, "mean", None)
                if isinstance(mean, torch.Tensor):
                    return mean
                if callable(mean):
                    return mean()
                if hasattr(latent_dist, "sample"):
                    return latent_dist.sample()

            if isinstance(encoded, tuple) and encoded:
                encoded = encoded[0]
                continue

            break

        raise TypeError(
            "Unsupported VAE encode output for SANA-WM first-frame conditioning: "
            f"{type(encoded).__name__}"
        )

    def _get_vae_scaling_factor(self, vae) -> float:
        scaling_factor = (
            getattr(getattr(vae, "config", None), "scaling_factor", None)
            or getattr(vae, "scaling_factor", None)
            or getattr(
                self.pipeline_config.vae_config.arch_config,
                "scaling_factor",
                None,
            )
            or 1.0
        )
        if isinstance(scaling_factor, torch.Tensor):
            return float(scaling_factor.item())
        scaling_factor = float(scaling_factor)
        return 1.0 if scaling_factor == 0.0 else scaling_factor

    def _prepare_noise_latents(
        self,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        generator: (
            torch.Generator | list[torch.Generator] | tuple[torch.Generator, ...]
        ),
    ) -> torch.Tensor:
        if isinstance(generator, (list, tuple)):
            if not generator:
                raise ValueError("SANA-WM generator list must not be empty.")
            if len(generator) == 1:
                return randn_tensor(
                    shape, generator=generator[0], device=device, dtype=dtype
                )
            if len(generator) != shape[0]:
                raise ValueError(
                    "SANA-WM generator list length must match latent batch size; "
                    f"got {len(generator)} generators for batch {shape[0]}."
                )
            sample_shape = (1, *shape[1:])
            return torch.cat(
                [
                    randn_tensor(
                        sample_shape,
                        generator=sample_generator,
                        device=device,
                        dtype=dtype,
                    )
                    for sample_generator in generator
                ],
                dim=0,
            )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    @staticmethod
    def _generator_from_seed(
        seed: int | list[int] | tuple[int, ...] | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Generator | list[torch.Generator]:
        if seed is None:
            seed = 0
        if isinstance(seed, (list, tuple)):
            if not seed:
                raise ValueError("SANA-WM seed list must not be empty.")
            if len(seed) == 1:
                seed = seed[0]
            elif len(seed) == batch_size:
                return [
                    torch.Generator(device=device).manual_seed(int(sample_seed))
                    for sample_seed in seed
                ]
            else:
                raise ValueError(
                    "SANA-WM seed list length must be 1 or match latent batch "
                    f"size; got {len(seed)} seeds for batch {batch_size}."
                )
        return torch.Generator(device=device).manual_seed(int(seed))

    @staticmethod
    def _splice_first_frame_latent(
        latents: torch.Tensor,
        first_frame_z: torch.Tensor,
        pipeline_config: Any | None = None,
    ) -> torch.Tensor:
        B = latents.shape[0]
        first_frame_z = first_frame_z.to(device=latents.device, dtype=latents.dtype)
        if first_frame_z.shape[0] == 1 and B > 1:
            first_frame_z = first_frame_z.expand(B, -1, -1, -1, -1)
        elif first_frame_z.shape[0] != B:
            raise ValueError(
                "SANA-WM first-frame latent batch does not match noise batch: "
                f"{first_frame_z.shape[0]} vs {B}."
            )

        latents = latents.clone()
        latents[:, :, 0:1] = first_frame_z
        log_sana_wm_tensor_stats(
            "latents.after_first_frame_splice",
            latents,
            pipeline_config,
        )
        return latents

    @torch.no_grad()
    def _splice_first_frame(
        self,
        latents: torch.Tensor,  # (B, 128, T_lat, H_sp, W_sp)
        condition_image,  # PIL Image or torch.Tensor
        dtype: torch.dtype,
        device: torch.device,
        batch: Req | None = None,
    ) -> torch.Tensor:
        """Replace latents[:, :, 0] with VAE-encoded first frame."""
        B, _C, _T_lat, H_sp, W_sp = latents.shape
        target_h = H_sp * self.pipeline_config.vae_stride[1]  # 32
        target_w = W_sp * self.pipeline_config.vae_stride[2]  # 32
        (
            first_frame_image_batch,
            preprocess_info_for_batch,
            preprocess_infos,
        ) = preprocess_sana_wm_condition_images_for_batch(
            condition_image,
            batch_size=B,
            target_h=target_h,
            target_w=target_w,
        )
        set_sana_wm_condition_image_preprocess_info(batch, preprocess_info_for_batch)
        self.log_info(
            "First-frame condition image preprocessed: source=%s, resized=%s, "
            "crop_offset=%s, target=%s.",
            preprocess_infos[0]["source_size"],
            preprocess_infos[0]["resized_size"],
            preprocess_infos[0]["crop_offset"],
            preprocess_infos[0]["target_size"],
        )
        if len(preprocess_infos) > 1:
            self.log_info(
                "Processed %d batched first-frame images.", len(preprocess_infos)
            )

        first_frame_z = self._vae_encode_image(first_frame_image_batch, dtype, device)
        return self._splice_first_frame_latent(
            latents,
            first_frame_z,
            self.pipeline_config,
        )

    def _prepare_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        device: torch.device,
    ):
        """Set up scheduler timesteps and populate batch.timesteps, .sigmas."""
        scheduler = get_or_create_request_scheduler(batch, self.scheduler)
        num_inference_steps = batch.num_inference_steps

        flow_shift = getattr(batch, "flow_shift", None)
        if flow_shift is None:
            flow_shift = getattr(
                self.pipeline_config,
                "inference_flow_shift",
                None,
            )
        if flow_shift is None:
            flow_shift = getattr(self.pipeline_config, "flow_shift", 9.95)
        flow_shift = float(flow_shift)
        kwargs = {}

        # diffusers FlowMatchEulerDiscreteScheduler supports mu/shift
        import inspect

        sig_params = inspect.signature(scheduler.set_timesteps).parameters
        if "shift" in sig_params:
            kwargs["shift"] = flow_shift
        elif "mu" in sig_params:
            # Convert flow_shift to mu: mu ≈ log(shift)
            import math

            kwargs["mu"] = math.log(flow_shift)

        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas.tolist()
        if sigmas:
            self.log_info(
                "FlowMatch timesteps prepared: steps=%d, flow_shift=%.4f, "
                "sigma_start=%.6f, sigma_end=%.6f",
                num_inference_steps,
                flow_shift,
                float(sigmas[0]),
                float(sigmas[-1]),
            )

        batch.timesteps = timesteps
        batch.sigmas = sigmas
        batch.scheduler = scheduler
        return batch

    def _adjust_num_frames_for_request(self, batch: Req) -> int:
        requested_num_frames = batch.num_frames or 49
        action_num_frames = sana_wm_action_num_frames_for_request(batch)
        if action_num_frames is not None and action_num_frames != requested_num_frames:
            self.log_info(
                "SANA-WM action trajectory has %d frames; keeping requested "
                "num_frames=%d and padding/trimming the action trajectory.",
                action_num_frames,
                requested_num_frames,
            )
        num_frames = self.pipeline_config.adjust_num_frames(requested_num_frames)
        if getattr(batch, "is_warmup", False):
            min_warmup_frames = int(self.pipeline_config.vae_stride[0]) + 1
            if num_frames < min_warmup_frames:
                self.log_info(
                    "SANA-WM warmup num_frames raised from %d to %d so temporal "
                    "stages receive at least two latent frames.",
                    num_frames,
                    min_warmup_frames,
                )
                return min_warmup_frames
        return num_frames

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """
        Pre-process everything needed by DenoisingStage for SANA-WM.

        Expects batch to already have prompt_embeds set by SanaWMTextEncodingStage.
        """
        device = get_local_torch_device()
        dtype = PRECISION_TO_TYPE.get(
            getattr(self.pipeline_config, "dit_precision", "bf16"),
            torch.bfloat16,
        )
        if not hasattr(batch, "extra") or batch.extra is None:
            batch.extra = {}

        # --- 0. Adjust num_frames to be compatible with VAE temporal stride ---
        num_frames = self._adjust_num_frames_for_request(batch)
        batch.num_frames = num_frames
        self.log_info(
            "SANA-WM prepare: seed=%s, size=%dx%d, frames=%d, "
            "vae_stride=%s, diagnostics=%s",
            getattr(batch, "seed", None),
            batch.width,
            batch.height,
            num_frames,
            self.pipeline_config.vae_stride,
            "on" if sana_wm_diagnostics_enabled(self.pipeline_config) else "off",
        )

        # --- 1. Generator for reproducibility ---
        batch_size = batch.batch_size or 1
        generator = getattr(batch, "generator", None)
        if not isinstance(generator, (list, tuple, torch.Generator)):
            generator = self._generator_from_seed(
                getattr(batch, "seed", None),
                batch_size=batch_size,
                device=device,
            )
            batch.generator = generator

        # --- 2. Compute latent shape and initialize noise ---
        latent_shape = self.pipeline_config.prepare_latent_shape(
            batch, batch_size, num_frames
        )
        # latent_shape: (B, 128, T_latent, H_sp, W_sp)
        latents = self._prepare_noise_latents(latent_shape, dtype, device, generator)
        log_sana_wm_tensor_stats(
            "latents.initial_noise",
            latents,
            self.pipeline_config,
        )

        # Store raw shape for DecodingStage
        batch.raw_latent_shape = latent_shape

        # --- 3. VAE-encode first frame and splice into noise latents ---
        condition_image = getattr(batch, "condition_image", None)
        if condition_image is not None:
            try:
                latents = self._splice_first_frame(
                    latents, condition_image, dtype, device, batch=batch
                )
                self.log_info("First-frame spliced into noise latents.")
            except Exception as e:
                raise RuntimeError(
                    "SANA-WM first-frame conditioning failed; refusing to "
                    "continue with pure-noise latents because that produces "
                    "misleading low-quality output."
                ) from e
        else:
            raise ValueError(
                "SANA-WM is a TI2V world model and requires condition_image "
                "for first-frame conditioning. Provide --image-path, "
                "--condition-image, or the equivalent API image input."
            )

        batch.latents = latents

        # --- 4. Camera conditioning ---
        # The released SANA-WM checkpoint is camera-conditioned. Native
        # inference requires a camera trajectory or action DSL. If the SGLang
        # request omits one, use a static identity trajectory so the UCPE path
        # remains active instead of silently dropping all camera conditioning.
        try:
            camera_conditions, chunk_plucker, camera_source = (
                SanaWMCameraConditioningBuilder(
                    self.pipeline_config,
                    log_info=self.log_info,
                ).build(
                    batch,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    latent_shape=latent_shape,
                    device=device,
                    dtype=dtype,
                )
            )
        except Exception as e:
            raise RuntimeError("SANA-WM camera conditioning failed.") from e

        if camera_conditions is not None:
            batch.extra["camera_conditions"] = camera_conditions
            log_sana_wm_tensor_stats(
                "camera_conditions",
                camera_conditions,
                self.pipeline_config,
            )
        if chunk_plucker is not None:
            batch.extra["chunk_plucker"] = chunk_plucker
            log_sana_wm_tensor_stats(
                "chunk_plucker",
                chunk_plucker,
                self.pipeline_config,
            )
        self.log_info(
            "SANA-WM camera conditioning: source=%s, raymap=%s, chunk_plucker=%s",
            camera_source,
            None if camera_conditions is None else tuple(camera_conditions.shape),
            None if chunk_plucker is None else tuple(chunk_plucker.shape),
        )

        # --- 5. Ensure prompt_embeds is a list (DenoisingStage expects list[Tensor]) ---
        if isinstance(batch.prompt_embeds, torch.Tensor):
            batch.prompt_embeds = [batch.prompt_embeds]
        if batch.negative_prompt_embeds is not None and isinstance(
            batch.negative_prompt_embeds, torch.Tensor
        ):
            batch.negative_prompt_embeds = [batch.negative_prompt_embeds]

        # --- 6. CFG setup ---
        batch.do_classifier_free_guidance = _sana_wm_should_do_cfg(batch)

        # --- 7. Prepare timesteps and sigmas ---
        batch = self._prepare_timesteps(batch, server_args, device)

        self.log_info(
            "BeforeDenoisingStage done: latent=%s, T_lat=%d, H_sp=%d, W_sp=%d, "
            "num_inference_steps=%d, camera=%s",
            str(latent_shape),
            latent_shape[2],
            latent_shape[3],
            latent_shape[4],
            batch.num_inference_steps,
            "yes" if batch.extra.get("camera_conditions") is not None else "no",
        )
        return batch
