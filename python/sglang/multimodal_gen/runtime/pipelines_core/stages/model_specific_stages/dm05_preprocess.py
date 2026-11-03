# SPDX-License-Identifier: Apache-2.0
"""DM05 observation transforms matching the Apache-2.0 OpenDM reference."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from sglang.multimodal_gen.runtime.vla.observation import VLAObservationBatch

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.pipeline_configs.dm05 import DM05PipelineConfig

_HISTORY_TOKENS_PER_IMAGE = 16
_G2_NATIVE_TO_SHARED = np.asarray(
    [0, 1, 2, 3, 4, 5, 20, 6, 7, 8, 9, 10, 11, 12, 21, 13]
)
_MANIPULATION_NATIVE_TO_SHARED = np.append(_G2_NATIVE_TO_SHARED, 14)


def _array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"DM05 {name} must contain numeric values") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"DM05 {name} must contain only finite values")
    return result


def _task_text(prompt: str) -> str:
    text = prompt.strip()
    if text and text[-1] not in ".!?。！？…":
        text += "."
    if text and not text.lower().startswith(("task:", "subtask:")):
        text = "Task: " + text
    return text


def normalize_dm05_vector(
    value: Any, stats: Mapping[str, Any], *, clip_to_bounds: bool = True
) -> np.ndarray:
    array = _array(value, "state")
    lo = np.asarray(stats["q01"], dtype=np.float32)
    hi = np.asarray(stats["q99"], dtype=np.float32)
    if array.shape[-1:] != lo.shape or lo.shape != hi.shape:
        raise ValueError(
            f"DM05 normalization expects final dimension {lo.size}, got {array.shape}"
        )
    if clip_to_bounds:
        array = np.clip(array, lo, hi)
    result = (array - lo) / (hi - lo + 1e-6) * 2.0 - 1.0
    return np.where((lo == 0) & (hi == 0), 0.0, result).astype(np.float32)


def denormalize_dm05_vector(value: Any, stats: Mapping[str, Any]) -> np.ndarray:
    array = _array(value, "actions")
    lo = np.asarray(stats["q01"], dtype=np.float32)
    hi = np.asarray(stats["q99"], dtype=np.float32)
    if array.shape[-1:] != lo.shape or lo.shape != hi.shape:
        raise ValueError(
            f"DM05 denormalization expects final dimension {lo.size}, got {array.shape}"
        )
    result = (array + 1.0) / 2.0 * (hi - lo + 1e-6) + lo
    return np.where((lo == 0) & (hi == 0), 0.0, result).astype(np.float32)


def dm05_state_bins(state: Any, n_bins: int = 256) -> list[int]:
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("DM05 n_state_bins must be an integer >= 2")
    clipped = np.clip(_array(state, "state"), -1.0, 1.0)
    result = np.floor((clipped + 1.0) / 2.0 * (n_bins - 1)).astype(int)
    return np.clip(result, 0, n_bins - 1).tolist()


class DM05NormStats:
    def __init__(self, payload: Mapping[str, Any]):
        self.default_robot_type = payload.get("default_robot_type")
        self.profiles = payload.get("norm_stats_by_robot")
        self.control_modes = payload.get("control_modes")
        self.default_profile = payload.get("norm_stats")
        if not isinstance(self.default_profile, Mapping):
            raise ValueError("DM05 norm_stats.json requires a norm_stats object")
        if self.control_modes is not None and not isinstance(
            self.control_modes, Mapping
        ):
            raise ValueError("DM05 normalization control_modes must be an object")
        if self.profiles is not None:
            if (
                not isinstance(self.profiles, Mapping)
                or not self.profiles
                or self.default_robot_type not in self.profiles
            ):
                raise ValueError("DM05 normalization requires a valid default robot")
        profiles = (
            self.profiles.values()
            if self.profiles is not None
            else [self.default_profile]
        )
        for profile in list(profiles) + list((self.control_modes or {}).values()):
            if not isinstance(profile, Mapping):
                raise ValueError("DM05 normalization profiles must be objects")
            for key in ("state", "action"):
                stats = profile.get(key)
                if not isinstance(stats, Mapping):
                    raise ValueError(f"DM05 normalization profile requires {key}")
                lo = _array(stats.get("q01"), f"{key} q01")
                hi = _array(stats.get("q99"), f"{key} q99")
                if lo.ndim != 1 or not lo.size or lo.shape != hi.shape:
                    raise ValueError(
                        f"DM05 {key} quantiles must be equal-length vectors"
                    )
                if np.any(hi < lo):
                    raise ValueError(f"DM05 {key} q99 must be >= q01")
        if self.profiles is not None:
            profile = self.profiles[self.default_robot_type]
            for key in ("state", "action"):
                for statistic in self.default_profile[key]:
                    if statistic not in profile[key] or not np.array_equal(
                        self.default_profile[key][statistic], profile[key][statistic]
                    ):
                        raise ValueError(
                            "DM05 norm_stats must match the default robot profile"
                        )

    def select(
        self, robot_type: str | None, control_mode: str | None = None
    ) -> Mapping[str, Any]:
        if self.profiles is None:
            profile = self.default_profile
        else:
            robot_type = robot_type or self.default_robot_type
            if robot_type not in self.profiles:
                raise ValueError(
                    f"DM05 normalization has no robot profile {robot_type!r}; "
                    f"available: {sorted(self.profiles)}"
                )
            profile = self.profiles[robot_type]
        if self.control_modes is not None and control_mode:
            mode = "eef" if control_mode in {"ee", "end effector"} else control_mode
            if mode not in self.control_modes:
                raise ValueError(
                    f"DM05 normalization has no control mode {control_mode!r}"
                )
            profile = self.control_modes[mode]
        return profile


def _rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    is_tensor = isinstance(value, torch.Tensor)
    if is_tensor:
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and is_tensor and array.shape[0] in (1, 3, 4):
        array = array.transpose(1, 2, 0)
    if array.ndim not in (2, 3) or (
        array.ndim == 3 and array.shape[-1] not in (1, 3, 4)
    ):
        raise ValueError(f"DM05 image must be RGB/HWC or CHW tensor, got {array.shape}")
    if array.dtype.kind not in "biuf":
        raise ValueError("DM05 image pixels must be real numeric values")
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 255):
        raise ValueError("DM05 input pixels must be finite values in [0,255]")
    if np.issubdtype(array.dtype, np.floating) and np.any(array != np.floor(array)):
        raise ValueError("DM05 raw pixels must be uint8 values; use encoded RGB images")
    array = array.astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array).convert("RGB")


def preprocess_dm05_image(
    value: Any, size: tuple[int, int] = (448, 448)
) -> Image.Image:
    import cv2

    image = np.asarray(_rgb_image(value))
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("DM05 image must not be empty")
    side = max(height, width)
    top = (side - height) // 2
    left = (side - width) // 2
    padded = cv2.copyMakeBorder(
        image,
        top,
        side - height - top,
        left,
        side - width - left,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    resized = cv2.resize(padded, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(resized, mode="RGB")


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    return quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)


def _rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float32)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    scale = np.where(
        angle > 1e-8,
        np.sin(0.5 * angle) / np.maximum(angle, 1e-8),
        0.5 - angle * angle / 48.0,
    )
    return _normalize_quaternion(
        np.concatenate([np.cos(0.5 * angle), rotvec * scale], axis=-1)
    ).astype(np.float32)


def _quaternion_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quaternion(np.asarray(quat, dtype=np.float32))
    quat = np.where(quat[..., :1] < 0, -quat, quat)
    vec = quat[..., 1:4]
    length = np.linalg.norm(vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(length, quat[..., :1])
    return np.where(
        length > 1e-8, vec * (angle / np.maximum(length, 1e-8)), np.zeros_like(vec)
    ).astype(np.float32)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = [left[..., i] for i in range(4)]
    rw, rx, ry, rz = [right[..., i] for i in range(4)]
    return _normalize_quaternion(
        np.stack(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            axis=-1,
        )
    ).astype(np.float32)


def _rpy_to_axis_angle(rpy: np.ndarray) -> np.ndarray:
    rpy = np.asarray(rpy, dtype=np.float32)
    roll, pitch, yaw = [rpy[..., i] * 0.5 for i in range(3)]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return _quaternion_to_rotvec(
        np.stack(
            [
                cy * cp * cr + sy * sp * sr,
                cy * cp * sr - sy * sp * cr,
                sy * cp * sr + cy * sp * cr,
                sy * cp * cr - cy * sp * sr,
            ],
            axis=-1,
        )
    )


def _state_description(
    robot_type: str | None, configured: tuple[str, ...] | list[str] | None, size: int
) -> list[str]:
    if configured is not None:
        description = [str(getattr(item, "value", item)) for item in configured]
    elif robot_type in {"DOS W1", "Aloha", "Aloha RoboTwin2"}:
        description = ["joint"] * 6 + ["gripper"] + ["joint"] * 6 + ["gripper"]
    elif robot_type == "Franka":
        description = ["joint"] * 6 + ["gripper"] * (size - 6)
    elif robot_type == "SO101":
        description = ["joint"] * 5 + ["gripper"]
    elif robot_type == "ARX5":
        description = ["joint"] * 6 + ["gripper"]
    elif robot_type == "UR5":
        description = ["eef"] * 6 + ["gripper"]
    elif robot_type is None:
        description = ["joint"] * size
    else:
        raise ValueError(f"DM05 requires a state description for robot {robot_type!r}")
    if len(description) != size:
        raise ValueError(
            f"DM05 robot state description has {len(description)} values; expected {size}"
        )
    return description


def _layout_state(state: np.ndarray, layout: str) -> tuple[np.ndarray, dict[str, Any]]:
    if layout == "identity":
        return state.copy(), {}
    if layout not in {"robocoliseum_g2", "robocoliseum_manipulation"}:
        raise ValueError(f"Unknown DM05 state_layout {layout!r}")
    expected = 16 if layout == "robocoliseum_g2" else 21
    if state.shape != (expected,):
        raise ValueError(f"DM05 {layout} requires native state dimension {expected}")
    shared = np.zeros(22, dtype=np.float32)
    indices = _G2_NATIVE_TO_SHARED if expected == 16 else _MANIPULATION_NATIVE_TO_SHARED
    predicted = state if expected == 16 else state[np.r_[np.arange(16), 20]]
    shared[indices] = predicted
    if expected == 21:
        shared[[6, 13]] *= -1
    valid = np.zeros(22, dtype=bool)
    valid[indices] = True
    if expected == 21:
        valid[[6, 13]] = False
    description = ["joint"] * 22
    description[6] = description[13] = "gripper"
    return shared, {
        "native_state": state.copy(),
        "valid_state_dimensions": valid,
        "state_desc": description,
    }


class DM05Preprocessor:
    def __init__(
        self,
        config: DM05PipelineConfig,
        model_path: str | None = None,
        *,
        processor: Any = None,
        norm_stats: Mapping[str, Any] | None = None,
    ):
        self.config = config
        source = getattr(config, "processor_name", None) or model_path
        if processor is None:
            if not source:
                raise ValueError(
                    "DM05 requires a processor_name or checkpoint model_path"
                )
            processor = AutoProcessor.from_pretrained(source)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        if norm_stats is None:
            norm_path = getattr(config, "norm_stats_path", None)
            if norm_path is None and model_path and Path(model_path).is_dir():
                norm_path = Path(model_path) / "norm_stats.json"
            norm_source = model_path or source
            if norm_path is None and norm_source and not Path(norm_source).exists():
                from huggingface_hub import hf_hub_download

                norm_path = hf_hub_download(norm_source, filename="norm_stats.json")
            if norm_path is None or not Path(norm_path).is_file():
                raise FileNotFoundError(
                    "DM05 requires checkpoint norm_stats.json or norm_stats_path"
                )
            norm_stats = json.loads(Path(norm_path).read_text())
        self.norm_stats = DM05NormStats(norm_stats)
        self.image_keys = tuple(config.image_keys)
        self.image_prompts = tuple(config.image_prompts)
        if len(self.image_keys) != len(self.image_prompts) or not self.image_keys:
            raise ValueError(
                "DM05 image_keys and image_prompts must have equal positive length"
            )
        profiles = (
            self.norm_stats.profiles.values()
            if self.norm_stats.profiles is not None
            else [self.norm_stats.default_profile]
        )
        for profile in list(profiles) + list(
            (self.norm_stats.control_modes or {}).values()
        ):
            if len(profile["action"]["q01"]) != config.output_action_dim:
                raise ValueError(
                    "DM05 output_action_dim must match every action norm profile"
                )

    def _messages(
        self,
        prompt: str,
        images: list[Image.Image],
        history: list[Image.Image],
        normalized_state: np.ndarray,
        meta: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        config = self.config
        layout = getattr(config, "state_layout", "identity")
        if layout.startswith("robocoliseum"):
            text = f"Robot: {meta['robot_type']}\nOverall speed: {meta['speed']}\n"
            if prompt:
                text += prompt + "\n"
        else:
            text = ""
            if meta["robot_type"] is not None:
                text += f"Robot: {meta['robot_type']}\n"
            if meta["control_mode"] is not None:
                mode = meta["control_mode"]
                mode = "end effector" if mode in {"eef", "ee", "end effector"} else mode
                text += f"Control mode: {mode}\n"
            text += f"Overall speed: {meta['speed']}\n"
            text += (
                prompt + "\n"
                if getattr(config, "prompt_is_formatted", False)
                and not getattr(config, "normalize_task_prompt", False)
                else f"Task: {prompt}.\n"
            )
        if getattr(config, "is_history", False):
            count = int(config.max_history_images)
            text += "History images: "
            text += "<unused1>" * (_HISTORY_TOKENS_PER_IMAGE * (count - len(history)))
            text += ("<unused0>" * _HISTORY_TOKENS_PER_IMAGE + "\n") * len(history)
        content = [{"type": "text", "text": text}]
        for label, image in zip(self.image_prompts, images, strict=True):
            text = f"{label} image: "
            if content[-1]["type"] == "text":
                content[-1]["text"] += text
            else:
                content.append({"type": "text", "text": text})
            content.append({"type": "image", "image": image})
        if config.add_state:
            if meta.get("valid_state_dimensions") is not None:
                normalized_state = normalized_state[meta["valid_state_dimensions"]]
            bins = dm05_state_bins(normalized_state, config.n_state_bins)
            content.append(
                {
                    "type": "text",
                    "text": "States: "
                    + " ".join(map(str, bins))
                    + ("\n" if layout.startswith("robocoliseum") else ""),
                }
            )
        return [{"role": "user", "content": content}]

    def _encode_messages(self, messages: list[dict[str, Any]], arguments):
        if getattr(self.config, "normalize_task_prompt", False):
            # The released MEM processor formats a copy at the encoding boundary.
            # Truncation must still count and replace the original raw prompt.
            messages = [
                {**message, "content": [dict(part) for part in message["content"]]}
                for message in messages
            ]
            content = messages[0]["content"][0]
            content["text"] = re.sub(
                r"(?ms)^Task: (.*?)\.\n(?=History images:)",
                lambda match: _task_text(match.group(1)) + "\n",
                content["text"],
            )
        return self.processor.apply_chat_template(messages, **arguments)

    def _tokenize(self, messages: list[dict[str, Any]], prompt: str):
        arguments = dict(
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = self._encode_messages(messages, arguments)
        max_length = int(self.config.max_token_len)
        if encoded["input_ids"].shape[1] > max_length:
            if getattr(self.config, "reject_prompt_truncation", False):
                raise ValueError(
                    f"DM05 prefix exceeds configured max_token_len={max_length}"
                )
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            overflow = encoded["input_ids"].shape[1] - max_length
            keep = max(0, len(prompt_tokens) - overflow - 16)
            if keep < len(prompt_tokens):
                shortened = self.tokenizer.decode(
                    prompt_tokens[:keep], skip_special_tokens=False
                ).strip()
                for item in messages[0]["content"]:
                    if item["type"] == "text" and prompt in item["text"]:
                        item["text"] = item["text"].replace(prompt, shortened, 1)
                        break
                if not getattr(self.config, "state_layout", "identity").startswith(
                    "robocoliseum"
                ):
                    arguments.pop("add_generation_prompt")
                encoded = self._encode_messages(messages, arguments)
            if encoded["input_ids"].shape[1] > max_length:
                raise ValueError(
                    f"DM05 fixed image/history prefix exceeds max_token_len={max_length}"
                )
        return encoded

    def __call__(self, raw_observation: dict[str, Any]) -> VLAObservationBatch:
        config = self.config
        if not isinstance(raw_observation, Mapping):
            raise ValueError("DM05 observation must be a mapping")
        mode = raw_observation.get("control_mode", config.default_control_mode)
        if mode is not None and not isinstance(mode, str):
            raise ValueError("DM05 control_mode must be a string or null")
        prompt = raw_observation.get("prompt", "")
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError("DM05 expects one prompt per action request")
            prompt = prompt[0]
        prompt = "" if prompt is None else str(prompt)
        requested_order = raw_observation.get("camera_order")
        if requested_order is not None and (
            not isinstance(requested_order, (list, tuple))
            or any(not isinstance(name, str) for name in requested_order)
        ):
            raise ValueError("DM05 camera_order must be a list of camera names")
        camera_order = (
            self.image_keys if requested_order is None else tuple(requested_order)
        )
        if camera_order != self.image_keys:
            raise ValueError(
                "DM05 camera_order must match the configured image_keys order"
            )
        values = raw_observation.get("images")
        if not isinstance(values, Mapping) or set(values) != set(camera_order):
            raise ValueError(
                f"DM05 requires exactly the configured views {camera_order}"
            )
        masks = raw_observation.get("image_masks")
        if masks is None:
            masks = {}
        if not isinstance(masks, Mapping) or any(
            not isinstance(name, str) or name not in camera_order for name in masks
        ):
            raise ValueError(
                "DM05 image_masks must map configured camera names to bools"
            )
        if any(not isinstance(value, (bool, np.bool_)) for value in masks.values()):
            raise ValueError("DM05 image_masks values must be boolean")
        if any(
            values[name] is None or not bool(masks.get(name, True))
            for name in camera_order
        ):
            raise ValueError(
                "DM05 requires all current camera views; masked cameras are unsupported"
            )
        images = [
            preprocess_dm05_image(values[name], config.image_size)
            for name in camera_order
        ]
        history_in = raw_observation.get("history_images")
        if history_in is None:
            history_in = []
        if not isinstance(history_in, (tuple, list)):
            raise ValueError("DM05 history_images must be a list")
        if history_in and not config.is_history:
            raise ValueError("DM05 history_images require is_history=true")
        if len(history_in) > config.max_history_images:
            raise ValueError(
                f"DM05 accepts at most {config.max_history_images} history images"
            )
        history = [
            preprocess_dm05_image(value, config.image_size) for value in history_in
        ]
        raw_state = raw_observation.get("state")
        if raw_state is None:
            raise ValueError("DM05 observation.state is required")
        raw_state = _array(raw_state, "state")
        if raw_state.ndim == 2 and raw_state.shape[0] == 1:
            raw_state = raw_state[0]
        if raw_state.ndim != 1 or not raw_state.size:
            raise ValueError("DM05 state must be a nonempty vector")
        layout = getattr(config, "state_layout", "identity")
        state, layout_meta = _layout_state(raw_state, layout)
        robot = raw_observation.get("robot_type")
        if robot is None:
            robot = config.default_robot_type or self.norm_stats.default_robot_type
        if robot is not None:
            robot = str(robot)
        if layout.startswith("robocoliseum"):
            if robot is not None and str(robot).strip().lower() not in {
                "agibot_g2",
                "agibot g2",
                "g2",
                "robocoliseum",
            }:
                raise ValueError(f"DM05 unsupported RoboColiseum robot {robot!r}")
            robot = "Agibot G2"
        elif (
            robot is not None
            and robot
            not in {
                "DOS W1",
                "Franka",
                "Aloha",
                "Aloha RoboTwin2",
                "SO101",
                "ARX5",
                "UR5",
            }
            and not (
                robot == config.default_robot_type
                and getattr(config, "state_desc", None) is not None
            )
        ):
            raise ValueError(f"DM05 unsupported robot_type {robot!r}")
        if layout.startswith("robocoliseum"):
            prompt = _task_text(prompt)
        speed = raw_observation.get("speed", config.default_speed)
        speed = str(speed or "0.5")
        profile = self.norm_stats.select(robot, mode)
        if state.shape != (len(profile["state"]["q01"]),):
            raise ValueError(
                f"DM05 state must have {len(profile['state']['q01'])} dimensions"
            )
        configured_description = getattr(config, "state_desc", None)
        if (
            config.use_absolute_action
            and robot is None
            and configured_description is None
            and not layout_meta.get("state_desc")
        ):
            raise ValueError("DM05 absolute actions require robot_type or state_desc")
        description = layout_meta.get("state_desc") or _state_description(
            robot, configured_description, state.size
        )
        transformed_state = state.copy()
        if config.compose_eef_rot and mode in {"eef", "ee", "end effector"}:
            if state.size not in (7, 14):
                raise ValueError(
                    "DM05 EEF state conversion requires 7 or 14 dimensions"
                )
            description = []
            for start in range(0, state.size, 7):
                transformed_state[start + 3 : start + 6] = _rpy_to_axis_angle(
                    state[start + 3 : start + 6]
                )
                description.extend(["eef"] * 6 + ["gripper"])
        normalized = normalize_dm05_vector(
            transformed_state, profile["state"], clip_to_bounds=config.clip_to_bounds
        )
        meta = {
            **layout_meta,
            "robot_type": robot,
            "control_mode": mode,
            "speed": speed,
            "state_desc": description,
        }
        messages = self._messages(prompt, images, history, normalized, meta)
        encoded = self._tokenize(messages, prompt)
        tokens = encoded["input_ids"].long()
        token_masks = encoded["attention_mask"].bool()
        pixels = encoded["pixel_values"]
        if (
            tokens.ndim != 2
            or tokens.shape[0] != 1
            or token_masks.shape != tokens.shape
        ):
            raise ValueError(
                "DM05 processor must produce one token sequence per request"
            )
        if pixels.shape[0] != len(camera_order):
            raise ValueError("DM05 processor must produce one crop per current camera")
        token_types = encoded["token_type_ids"].long()
        history_pixels = None
        history_mask = None
        if config.is_history:
            placeholder = self.tokenizer.convert_tokens_to_ids("<unused0>")
            history_mask = tokens == placeholder
            if int(history_mask.sum()) != len(history) * _HISTORY_TOKENS_PER_IMAGE:
                raise ValueError(
                    "DM05 history placeholder count does not match history images"
                )
            token_types = token_types.clone()
            token_types[history_mask] = 1
            if history:
                history_pixels = self.processor.image_processor(
                    images=history, return_tensors="pt"
                )["pixel_values"]
        action_mask = torch.zeros(1, 1, config.action_dim, dtype=torch.float32)
        action_mask[
            ...,
            : (
                config.action_dim
                if config.full_action_mask
                else config.output_action_dim
            ),
        ] = 1
        noise = raw_observation.get("noise")
        if noise is not None:
            noise = torch.from_numpy(_array(noise, "noise"))
            if noise.ndim == 2:
                noise = noise.unsqueeze(0)
            expected = (1, config.action_horizon, config.action_dim)
            if tuple(noise.shape) != expected:
                raise ValueError(
                    f"DM05 noise must have shape {expected}, got {tuple(noise.shape)}"
                )
        postprocess = {
            "profile": profile,
            "state": normalized if config.use_transformed_state else state,
            "state_desc": description,
            "native_state": layout_meta.get("native_state"),
            "layout": layout,
        }
        return VLAObservationBatch(
            prompt=[prompt],
            images={
                name: pixels[index : index + 1]
                for index, name in enumerate(camera_order)
            },
            image_masks={name: torch.tensor([True]) for name in camera_order},
            state=torch.from_numpy(normalized.copy()).unsqueeze(0),
            noise=noise,
            tokens=tokens,
            token_masks=token_masks,
            batch_size=1,
            metadata={
                "camera_order": camera_order,
                "dm05": {
                    "pixel_values": pixels,
                    "token_type_ids": token_types,
                    "history_pixel_values": history_pixels,
                    "history_mask": history_mask,
                    "action_mask": action_mask,
                },
                "dm05_postprocess": [postprocess],
            },
        )


def postprocess_dm05_actions(
    actions: Any, observation: VLAObservationBatch, config: DM05PipelineConfig
) -> np.ndarray:
    result = _array(actions, "actions")
    if result.ndim == 2:
        result = result[None]
    if result.ndim != 3 or result.shape[0] != observation.batch_size:
        raise ValueError("DM05 actions must contain one action chunk per observation")
    contexts = observation.metadata["dm05_postprocess"]
    if len(contexts) != result.shape[0]:
        raise ValueError("DM05 postprocessing contexts do not match action batch")
    outputs = []
    for normalized, context in zip(result, contexts, strict=True):
        profile = context["profile"]
        action = denormalize_dm05_vector(
            normalized[..., : config.output_action_dim], profile["action"]
        )
        state = np.asarray(context["state"], dtype=np.float32)
        if config.denorm_state:
            state = denormalize_dm05_vector(state, profile["state"])
        if config.use_absolute_action:
            if state.shape[-1] != action.shape[-1]:
                raise ValueError(
                    "DM05 relative reconstruction requires matching state/action dimensions"
                )
            absolute = state + action
            descriptions = context["state_desc"]
            grippers = [i for i, name in enumerate(descriptions) if name == "gripper"]
            absolute[..., grippers] = action[..., grippers]
            if config.compose_eef_rot:
                start = 0
                while start < len(descriptions):
                    if descriptions[start] != "eef":
                        start += 1
                        continue
                    stop = start
                    while stop < len(descriptions) and descriptions[stop] == "eef":
                        stop += 1
                    if stop - start >= 6:
                        rotation = slice(start + 3, start + 6)
                        absolute[..., rotation] = _quaternion_to_rotvec(
                            _quaternion_multiply(
                                _rotvec_to_quaternion(action[..., rotation]),
                                _rotvec_to_quaternion(state[rotation]),
                            )
                        )
                    start = stop
            action = absolute
        layout = context["layout"]
        if layout == "robocoliseum_g2":
            action = action[..., _G2_NATIVE_TO_SHARED]
        elif layout == "robocoliseum_manipulation":
            native = np.empty(action.shape[:-1] + (21,), dtype=np.float32)
            native[..., np.r_[np.arange(16), 20]] = action[
                ..., _MANIPULATION_NATIVE_TO_SHARED
            ]
            native[..., [7, 15]] *= -1
            native[..., 16:20] = context["native_state"][16:20]
            action = native
        outputs.append(action.astype(np.float32, copy=False))
    return np.stack(outputs)


def dm05_observation_signature(observation: VLAObservationBatch) -> tuple[Any, ...]:
    fields = observation.metadata["dm05"]
    current = fields["pixel_values"]
    history = fields.get("history_pixel_values")
    action_mask = fields["action_mask"]
    token_types = fields["token_type_ids"]
    history_mask = fields.get("history_mask")
    contexts = observation.metadata["dm05_postprocess"]
    return (
        tuple(observation.metadata["camera_order"]),
        (current.shape[0] // observation.batch_size, *current.shape[1:]),
        current.dtype,
        (
            None
            if history_mask is None
            else (
                tuple(current.shape[1:])
                if history is None
                else tuple(history.shape[1:]),
                current.dtype if history is None else history.dtype,
            )
        ),
        tuple(observation.tokens.shape[1:]),
        observation.tokens.dtype,
        tuple(token_types.shape[1:]),
        token_types.dtype,
        None
        if history_mask is None
        else (tuple(history_mask.shape[1:]), history_mask.dtype),
        tuple(action_mask.shape[1:]),
        action_mask.dtype,
        None if observation.state is None else tuple(observation.state.shape[1:]),
        None
        if observation.noise is None
        else (
            tuple(observation.noise.shape[1:]),
            observation.noise.dtype,
            observation.noise.device,
        ),
        tuple(sorted({context["layout"] for context in contexts})),
    )


def collate_dm05_observations(
    observations: list[VLAObservationBatch],
) -> VLAObservationBatch:
    if not observations:
        raise ValueError("DM05 cannot collate an empty observation list")
    first = observations[0]
    signature = dm05_observation_signature(first)
    if any(dm05_observation_signature(obs) != signature for obs in observations[1:]):
        raise ValueError("DM05 observations have incompatible execution layouts")
    order = tuple(first.metadata["camera_order"])

    def concatenate_optional(values):
        present = [value for value in values if value is not None]
        return torch.cat(present, dim=0) if present else None

    fields = [obs.metadata["dm05"] for obs in observations]
    merged_fields = {
        name: concatenate_optional([field.get(name) for field in fields])
        for name in (
            "pixel_values",
            "token_type_ids",
            "history_pixel_values",
            "history_mask",
            "action_mask",
        )
    }
    return VLAObservationBatch(
        prompt=[prompt for obs in observations for prompt in obs.prompt],
        images={
            name: torch.cat([obs.images[name] for obs in observations], dim=0)
            for name in order
        },
        image_masks={
            name: torch.cat([obs.image_masks[name] for obs in observations], dim=0)
            for name in order
        },
        state=concatenate_optional([obs.state for obs in observations]),
        noise=concatenate_optional([obs.noise for obs in observations]),
        tokens=torch.cat([obs.tokens for obs in observations], dim=0),
        token_masks=torch.cat([obs.token_masks for obs in observations], dim=0),
        batch_size=sum(obs.batch_size for obs in observations),
        metadata={
            "camera_order": order,
            "dm05": merged_fields,
            "dm05_postprocess": [
                context
                for obs in observations
                for context in obs.metadata["dm05_postprocess"]
            ],
        },
    )
