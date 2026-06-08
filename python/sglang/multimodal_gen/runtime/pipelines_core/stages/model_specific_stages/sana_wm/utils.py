# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Small utility surface for standalone SANA-WM inference."""

from __future__ import annotations

import gc
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import IO, Any

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

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
MIN_FOV_DEG = 25.0
MAX_FOV_DEG = 120.0
ALLOWED_ACTION_KEYS = frozenset("wasdijkl")


def split_hf_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("hf://"):
        raise ValueError(f"not an hf:// URI: {uri}")
    parts = uri[len("hf://") :].split("/")
    if len(parts) < 2:
        raise ValueError(f"hf:// URI must include owner/repo: {uri}")
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def join_hf_or_local(root: str | Path, *parts: str) -> str:
    root = str(root)
    suffix = "/".join(part.strip("/") for part in parts if part)
    if root.startswith("hf://"):
        return "/".join(part for part in (root.rstrip("/"), suffix) if part)
    return str(Path(root).joinpath(*parts))


def resolve_path(path: str | Path) -> str:
    path = str(path)
    if not path.startswith("hf://"):
        return path

    from huggingface_hub import hf_hub_download

    repo, filename = split_hf_uri(path)
    if not filename:
        raise ValueError(f"hf:// file URI must include a file path: {path}")
    return hf_hub_download(repo, filename)


def resolve_hf_dir(path: str | Path) -> str:
    path = str(path)
    if not path.startswith("hf://"):
        return path

    from huggingface_hub import snapshot_download

    repo, subdir = split_hf_uri(path)
    root = snapshot_download(repo, allow_patterns=f"{subdir}/**" if subdir else None)
    return str(Path(root) / subdir) if subdir else str(root)


def read_state_dict(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device=str(map_location))
    else:
        state = torch.load(path, map_location=map_location)

    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint {path} did not contain a state dict")
    state = {
        (key[len("model.") :] if key.startswith("model.") else key): value
        for key, value in state.items()
    }
    state.pop("pos_embed", None)
    return state


def load_checkpoint_fail_fast(
    model: torch.nn.Module,
    path: str | Path,
    *,
    allowed_missing: set[str] | None = None,
) -> None:
    allowed_missing = allowed_missing or {"pos_embed"}
    state = read_state_dict(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = sorted(key for key in missing if key not in allowed_missing)
    bad_unexpected = sorted(unexpected)
    if bad_missing or bad_unexpected:
        details = []
        if bad_missing:
            details.append(f"missing={bad_missing}")
        if bad_unexpected:
            details.append(f"unexpected={bad_unexpected}")
        raise RuntimeError(f"checkpoint does not match model: {'; '.join(details)}")


def pil_to_model_tensor(
    image: Image.Image, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor * 2.0 - 1.0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)


def write_video(path: Path, video: np.ndarray, fps: int) -> None:
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, video, fps=fps)


def _resolve_ffmpeg_binary() -> str:
    configured = os.environ.get("FFMPEG_BINARY")
    if configured:
        return configured
    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is required for streaming MP4 output; set FFMPEG_BINARY to a valid ffmpeg executable"
        ) from exc


class StreamingMp4Writer:
    def __init__(
        self,
        path: Path,
        *,
        height: int,
        width: int,
        fps: int,
        crf: int = 18,
        preset: str = "medium",
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames_written = 0
        ffmpeg = _resolve_ffmpeg_binary()
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            str(int(fps)),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            str(preset),
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._closed = False

    def write_chunk(self, frames_uint8: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        if (
            frames_uint8.dtype != np.uint8
            or frames_uint8.ndim != 4
            or frames_uint8.shape[-1] != 3
        ):
            raise ValueError(
                f"expected uint8 frames shaped (T,H,W,3), got {frames_uint8.shape} {frames_uint8.dtype}"
            )
        if not frames_uint8.flags["C_CONTIGUOUS"]:
            frames_uint8 = np.ascontiguousarray(frames_uint8)
        stdin: IO[bytes] | None = self._proc.stdin
        if stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        try:
            stdin.write(frames_uint8.tobytes())
        except BrokenPipeError as exc:
            stderr = (
                self._proc.stderr.read().decode(errors="replace")
                if self._proc.stderr is not None
                else ""
            )
            raise RuntimeError(f"ffmpeg exited while writing:\n{stderr}") from exc
        self.frames_written += int(frames_uint8.shape[0])

    def close(self) -> Path:
        if self._closed:
            return self.path
        self._closed = True
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        stderr = (
            self._proc.stderr.read().decode(errors="replace")
            if self._proc.stderr is not None
            else ""
        )
        ret = self._proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}:\n{stderr}")
        return self.path


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


def estimate_intrinsics_with_pi3x(
    image: Image.Image, device: torch.device | str = "cuda"
) -> np.ndarray:
    """Estimate ``[fx, fy, cx, cy]`` with Pi3X when --intrinsics is omitted."""
    try:
        from pi3.models.pi3x import Pi3X
        from pi3.utils.geometry import recover_intrinsic_from_rays_d
    except ImportError as exc:
        raise RuntimeError(
            "intrinsics were omitted, but Pi3X is not installed; pass --intrinsics or install pi3"
        ) from exc

    device = torch.device(device)
    orig_w, orig_h = image.size
    pixel_limit = 255_000
    scale = (
        math.sqrt(pixel_limit / (orig_w * orig_h))
        if orig_w * orig_h > pixel_limit
        else 1.0
    )
    target_w, target_h = orig_w * scale, orig_h * scale
    k = max(1, round(target_w / 14))
    m = max(1, round(target_h / 14))
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > target_w / target_h:
            k -= 1
        else:
            m -= 1
    model_w, model_h = max(1, k) * 14, max(1, m) * 14
    resized = image.resize((model_w, model_h), Image.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)

    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    model.disable_multimodal()
    model.requires_grad_(False)
    with torch.no_grad(), torch.amp.autocast(
        device.type, dtype=dtype, enabled=device.type == "cuda"
    ):
        out = model(imgs=tensor)
    rays_d = torch.nn.functional.normalize(out["local_points"], dim=-1)
    intrinsics = recover_intrinsic_from_rays_d(
        rays_d, force_center_principal_point=True
    )[0, 0]
    intrinsics = intrinsics.detach().cpu().float().numpy()

    sx, sy = orig_w / model_w, orig_h / model_h
    fx, fy = float(intrinsics[0, 0] * sx), float(intrinsics[1, 1] * sy)
    cx, cy = float(intrinsics[0, 2] * sx), float(intrinsics[1, 2] * sy)
    fov_x = math.degrees(2.0 * math.atan(orig_w / (2.0 * fx)))
    fov_y = math.degrees(2.0 * math.atan(orig_h / (2.0 * fy)))
    if not (MIN_FOV_DEG < fov_x < MAX_FOV_DEG and MIN_FOV_DEG < fov_y < MAX_FOV_DEG):
        raise RuntimeError(
            f"Pi3X-estimated FOV H={fov_x:.1f}, V={fov_y:.1f} is outside "
            f"[{MIN_FOV_DEG}, {MAX_FOV_DEG}]; pass trusted --intrinsics"
        )

    del model, out, rays_d, tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return np.array([fx, fy, cx, cy], dtype=np.float32)


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
