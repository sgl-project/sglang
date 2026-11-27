# SPDX-License-Identifier: Apache-2.0
"""Preserve raw RGB clips and robot coordinates for the native Motubrain2 processor."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.configs.sample.motubrain2 import (
    Motubrain2Observation,
    validate_observation_options,
)


def _as_float_array(value: Any, name: str) -> np.ndarray:
    """Normalize numpy / torch / SGLang tensor payloads to a float array."""

    if isinstance(value, dict):
        shape = value.get("shape")
        if "values" in value:
            value = value["values"]
        elif "data" in value:
            value = value["data"]
        else:
            raise ValueError(f"{name} dict payload must contain 'values' or 'data'")
        if shape is not None:
            value = np.asarray(value).reshape(tuple(shape))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def _extract_images(
    observation: dict[str, Any],
    camera_order: tuple[str, ...],
    alias_targets: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    images = observation.get("images") or observation.get("cameras")
    if images is None:
        images = {}
    if not isinstance(images, dict):
        raise ValueError(
            "observation.images must be a mapping from camera name to image"
        )

    result: dict[str, Any] = {}
    for name in camera_order:
        candidates = [name]
        for alias in (alias_targets or {}).get(name, ()):
            if alias not in candidates:
                candidates.append(alias)
        value = None
        for candidate in candidates:
            value = images.get(candidate)
            if value is None:
                value = observation.get(candidate)
            if value is None:
                value = observation.get(f"observation.images.{candidate}")
            if value is not None:
                break
        if value is None:
            raise ValueError(
                f"missing camera {name!r}; tried {candidates} under "
                "observation.images, flat keys, and observation.images.<name>"
            )
        result[name] = value
    return result


def _raw_frame_to_hwc_uint8(value: Any) -> np.ndarray:
    """Normalize one raw camera frame to an HWC uint8 array.

    The real Motubrain2 policy owns its resize/crop/normalize pipeline, so the
    SGLang preprocessor must not convert frames to float CHW here. Accept the
    same inputs clients already send to the action API: numpy/torch arrays,
    PIL images, CHW or HWC, uint8 or float in ``[0, 1]``/``[0, 255]``.
    """

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "convert"):
        value = np.asarray(value.convert("RGB"))
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError("camera frames must be HWC or CHW arrays")
    if array.shape[-1] == 3:
        hwc = array
    elif array.shape[0] == 3:
        hwc = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError("camera frames need three RGB channels")
    if min(hwc.shape) < 1:
        raise ValueError("camera frames must contain pixels")
    if hwc.dtype == np.uint8:
        return np.ascontiguousarray(hwc)
    if np.issubdtype(hwc.dtype, np.floating):
        if not np.isfinite(hwc).all():
            raise ValueError("camera frames must be finite")
        if hwc.size and float(hwc.max()) <= 1.0:
            hwc = hwc * 255.0
    hwc = np.clip(np.rint(hwc.astype(np.float64)), 0, 255)
    return np.ascontiguousarray(hwc.astype(np.uint8))


def _raw_camera_clip(value: Any, key: str) -> torch.Tensor:
    """Return one camera observation as a uint8 ``[T, H, W, 3]`` tensor."""

    # Nested frame payloads (especially sparse history) bypass the generic
    # action protocol's single-image conversion.
    if isinstance(value, dict) or isinstance(value, list) and value:
        from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
            _normalize_image_value,
        )

        if isinstance(value, list) and isinstance(value[0], dict):
            value = [_normalize_image_value(frame) for frame in value]
        else:
            value = _normalize_image_value(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
    if isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim == 4:
        value = [value[index] for index in range(value.shape[0])]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"camera {key!r} has no frames")
        frames = [_raw_frame_to_hwc_uint8(frame) for frame in value]
        shapes = {frame.shape for frame in frames}
        if len(shapes) != 1:
            raise ValueError(f"camera {key!r} frames must share one shape")
        clip = np.stack(frames)
    else:
        clip = _raw_frame_to_hwc_uint8(value)[None]
    # PIL may expose a read-only buffer; torch tensors require writable storage.
    return torch.from_numpy(np.require(clip, requirements=["C", "W"]))


def _pose_vector(value: Any, name: str) -> torch.Tensor:
    """Convert a state/extrinsic payload to a finite float32 CPU tensor."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
    else:
        tensor = torch.from_numpy(_as_float_array(value, name))
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


class Motubrain2Preprocessor:
    def __init__(self, config: Any):
        self.config = config
        self.camera_order = tuple(config.image_keys)
        self.camera_aliases = dict(getattr(config, "camera_aliases", {}) or {})
        self.alias_targets: dict[str, list[str]] = {}
        for alias, target in self.camera_aliases.items():
            self.alias_targets.setdefault(target, []).append(alias)

    def _raw_history_clip(
        self,
        past: dict[str, Any],
        source: str,
        frames: int,
    ) -> torch.Tensor:
        candidates = [source, *self.alias_targets.get(source, ())]
        value = None
        for candidate in candidates:
            value = past.get(candidate)
            if value is not None:
                break
        if value is None:
            raise ValueError(f"missing history camera {source!r}; tried {candidates}")
        clip = _raw_camera_clip(value, source)
        if clip.shape[0] != frames:
            raise ValueError(
                f"history camera {source!r} must carry {frames} RGB frames, "
                f"got {clip.shape[0]}"
            )
        return clip

    def __call__(self, observation: dict[str, Any]) -> Motubrain2Observation:
        rtc = validate_observation_options(observation, self.config)

        prompt = observation.get("prompt") or observation.get("task") or ""
        if not prompt:
            prompt = observation.get("instruction", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "Motubrain2 policy requires a nonempty 'prompt' or 'instruction'"
            )

        camera_order = self.camera_order
        raw = _extract_images(observation, camera_order, self.alias_targets)
        observed_frames = int(getattr(self.config, "observed_frames", 1))
        temporal_stride = int(getattr(self.config, "temporal_stride", 1))
        expected_frames = 1 + temporal_stride * (observed_frames - 1)
        cameras: dict[str, torch.Tensor] = {}
        for source in camera_order:
            clip = _raw_camera_clip(raw[source], source)
            if clip.shape[0] != expected_frames:
                raise ValueError(
                    f"camera {source!r} must carry {expected_frames} RGB frames "
                    f"for observed_frames={observed_frames}, "
                    f"temporal_stride={temporal_stride}; got {clip.shape[0]}"
                )
            cameras[source] = clip

        native: dict[str, Any] = {"instruction": prompt, "cameras": cameras}

        state = observation.get("state")
        if state is None:
            state = observation.get("observation.state")
        state_tensor = None
        if state is not None:
            state_tensor = _pose_vector(state, "state").reshape(-1)
            if state_tensor.numel() != self.config.state_dim:
                raise ValueError(f"state must contain {self.config.state_dim} values")
            native["state"] = state_tensor

        extrinsic = observation.get("camera_extrinsic")
        if extrinsic is not None:
            extrinsic = _pose_vector(extrinsic, "camera_extrinsic")
            if tuple(extrinsic.shape) != (4, 4):
                raise ValueError(
                    "camera_extrinsic must be a 4x4 base-from-camera matrix"
                )
            native["camera_extrinsic"] = extrinsic

        history_frames = getattr(self.config, "sparse_history_frames", None)
        if history_frames is not None:
            past = observation.get("history_cameras")
            if not isinstance(past, dict):
                raise ValueError(
                    "this profile uses sparse history; provide 'history_cameras'"
                )
            native["history_cameras"] = {
                source: self._raw_history_clip(past, source, int(history_frames))
                for source in camera_order
            }
        elif observation.get("history_cameras") is not None:
            raise ValueError(
                "history_cameras was supplied but the profile does not use "
                "sparse history"
            )

        return Motubrain2Observation(
            inputs=native,
            return_native_outputs=observation.get("return_native_outputs", False),
            session_id=observation.get("session_id"),
            reset=observation.get("reset", False),
            rtc=rtc,
            seed=observation.get("_motubrain2_seed", observation.get("seed")),
        )
