# SPDX-License-Identifier: Apache-2.0
"""LTX-2 stage-1 denoiser-output cache policies.

This module adapts the public dev_cache_core Stage1CacheController to the
SGLang LTX-2 denoising loop. It caches the denoiser-level video/audio outputs,
not individual transformer blocks.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


FLUX_TEACACHE_COEFFICIENTS = (
    4.98651651e2,
    -2.83781631e2,
    5.58554382e1,
    -3.82021401,
    2.64230861e-1,
)


_PRESETS: dict[str, dict[str, Any]] = {
    # SGLang HQ stage 1 uses 15 diffusion steps with res2s midpoint calls:
    # 14 steps have two denoiser calls and the last step has one, so the
    # runtime sees 29 denoiser-level calls. These presets map the public
    # dev_cache_core HQ winner schedules onto that 29-call topology while
    # preserving a final real denoiser call.
    "12of15_delta05_29calls": {
        "expected_calls": 29,
        "method": "fixedskip",
        "params": {
            "reuse_mode": "delta",
            "delta_scale": 0.5,
            "skip_indices": [22, 23, 24, 25, 26, 27],
            "max_skip_steps": 6,
            "start_percent": 0.0,
            "end_percent": 0.999,
        },
    },
    "10of15_last_29calls": {
        "expected_calls": 29,
        "method": "fixedskip",
        "params": {
            "reuse_mode": "last",
            "skip_indices": [19, 20, 21, 22, 23, 24, 25, 26, 27],
            "max_skip_steps": 9,
            "start_percent": 0.0,
            "end_percent": 0.999,
        },
    },
    "8of15_last_29calls": {
        "expected_calls": 29,
        "method": "fixedskip",
        "params": {
            "reuse_mode": "last",
            "skip_indices": [13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27],
            "max_skip_steps": 30,
            "start_percent": 0.0,
            "end_percent": 0.999,
        },
    },
    "5of15_blend_ema_29calls": {
        "expected_calls": 29,
        "method": "fixedskip",
        "params": {
            "reuse_mode": "blend_ema_last",
            "ema_decay": 0.7,
            "blend_alpha": 0.5,
            "skip_indices": [
                8,
                9,
                10,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
            ],
            "max_skip_steps": 30,
            "start_percent": 0.0,
            "end_percent": 0.999,
        },
    },
    # Exact public dev_cache_core schedules for the official 31-call HQ runner.
    "12of15_delta05_official31": {
        "expected_calls": 31,
        "method": "fixedskip",
        "params": {
            "reuse_mode": "delta",
            "delta_scale": 0.5,
            "skip_indices": [24, 25, 26, 27, 28, 29],
            "max_skip_steps": 6,
            "start_percent": 0.0,
            "end_percent": 0.999,
        },
    },
    "10of15_last_official31": {
        "expected_calls": 31,
        "method": "fixedskip",
        "params": {
            "reuse_mode": "last",
            "skip_indices": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "max_skip_steps": 10,
            "start_percent": 0.0,
            "end_percent": 0.999,
        },
    },
}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using %s", name, value, default)
        return default


def _env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value in (None, ""):
        return list(default)
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        logger.warning("Invalid integer list for %s=%r; using %s", name, value, default)
        return list(default)


def _poly1d(coefficients: tuple[float, ...], x: float) -> float:
    result = 0.0
    degree = len(coefficients) - 1
    for index, coeff in enumerate(coefficients):
        result += coeff * (x ** (degree - index))
    return result


def _clone_tensor(
    tensor: torch.Tensor | None, *, device: str = "default"
) -> torch.Tensor | None:
    if tensor is None:
        return None
    tensor = tensor.detach()
    if device == "cpu":
        return tensor.float().cpu()
    return tensor.clone()


def _to_device_dtype(
    tensor: torch.Tensor | None, like: torch.Tensor
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.to(device=like.device, dtype=like.dtype)


def _subsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    if factor > 1 and x.ndim >= 4:
        return x[..., ::factor, ::factor]
    return x


def _mean_abs(x: torch.Tensor) -> torch.Tensor:
    return x.float().abs().mean()


class LTX2Stage1CacheCoreController:
    """Stage-1 denoiser cache controller from dev_cache_core."""

    def __init__(
        self, method: str, params: dict[str, Any], *, expected_calls: int
    ) -> None:
        self.method = method
        self.params = dict(params)
        self.expected_calls = expected_calls
        self.total_calls = 0
        self.real_calls = 0
        self.skipped_calls = 0
        self.events: list[dict[str, Any]] = []
        self.last_x: torch.Tensor | None = None
        self.last_video_output: torch.Tensor | None = None
        self.last_audio_output: torch.Tensor | None = None
        self.previous_video_output: torch.Tensor | None = None
        self.previous_audio_output: torch.Tensor | None = None
        self.ema_video_output: torch.Tensor | None = None
        self.ema_audio_output: torch.Tensor | None = None
        self.cache_video_diff: torch.Tensor | None = None
        self.cache_audio_diff: torch.Tensor | None = None
        self.output_prev_subsampled: torch.Tensor | None = None
        self.x_prev_subsampled: torch.Tensor | None = None
        self.output_prev_norm: torch.Tensor | None = None
        self.relative_transformation_rate: torch.Tensor | None = None
        self.cumulative_change_rate = 0.0
        self.accumulated_rel_l1_distance = 0.0
        self.skipped_in_a_row = 0

    @property
    def max_skip_steps(self) -> int:
        return int(self.params.get("max_skip_steps", 1))

    @property
    def start_percent(self) -> float:
        return float(self.params.get("start_percent", 0.0))

    @property
    def end_percent(self) -> float:
        return float(self.params.get("end_percent", 1.0))

    @property
    def progress(self) -> float:
        if self.expected_calls <= 1:
            return 1.0
        return min(max((self.total_calls - 1) / (self.expected_calls - 1), 0.0), 1.0)

    def stats_summary(self) -> dict[str, Any]:
        eligible = self.real_calls + self.skipped_calls
        return {
            "method": self.method,
            "expected_calls": self.expected_calls,
            "total_calls": self.total_calls,
            "real_calls": self.real_calls,
            "skipped_calls": self.skipped_calls,
            "hit_rate": 0.0 if eligible == 0 else round(self.skipped_calls / eligible, 4),
            "reuse_mode": self.params.get("reuse_mode"),
            "skip_indices": list(self.params.get("skip_indices", [])),
            "events": list(self.events),
        }

    def _in_window(self) -> bool:
        progress = self.progress
        return self.start_percent <= progress < self.end_percent

    def _can_skip_common(self) -> bool:
        if self.total_calls <= 1:
            return False
        if self.total_calls >= self.expected_calls:
            return False
        if not self._in_window():
            return False
        if self.skipped_in_a_row >= self.max_skip_steps:
            return False
        return self.last_video_output is not None

    def _schedule_skip(self) -> bool:
        warmup = int(self.params.get("warmup_steps", 1))
        interval = max(int(self.params.get("interval", 1)), 1)
        if self.total_calls <= warmup:
            return False
        return (self.total_calls - warmup - 1) % interval == 0

    def should_skip(self, x: torch.Tensor) -> bool:
        self.total_calls += 1
        if not self._can_skip_common():
            return False

        if self.method == "fixedskip":
            skip_indices = {int(index) for index in self.params.get("skip_indices", [])}
            return (self.total_calls - 1) in skip_indices

        if self.method in {"periodic", "delta", "ema"}:
            return self._schedule_skip()

        if self.method == "similarity":
            warmup = int(self.params.get("warmup_steps", 1))
            if self.total_calls <= warmup or self.last_x is None:
                return False
            factor = int(self.params.get("subsample_factor", 8))
            current = _subsample(x.detach(), factor).float()
            previous = _subsample(self.last_x, factor).float().to(current.device)
            rel_change = float(
                (current - previous)
                .abs()
                .mean()
                .div(previous.abs().mean().clamp_min(1e-6))
                .item()
            )
            return rel_change < float(self.params.get("similarity_threshold", 0.05))

        if self.method == "teacache":
            cache_device = str(self.params.get("cache_device", "default"))
            current = _clone_tensor(x, device=cache_device)
            if self.last_x is None or self.cache_video_diff is None or current is None:
                self.last_x = current
                self.accumulated_rel_l1_distance = 0.0
                return False
            previous = self.last_x.float().to(current.device)
            rel_l1 = float(
                (current.float() - previous)
                .abs()
                .mean()
                .div(previous.abs().mean().clamp_min(1e-6))
                .item()
            )
            self.accumulated_rel_l1_distance += abs(
                _poly1d(FLUX_TEACACHE_COEFFICIENTS, rel_l1)
            )
            self.last_x = current
            return self.accumulated_rel_l1_distance < float(
                self.params.get("rel_l1_thresh", 0.4)
            )

        if self.method in {"easycache", "lazycache"}:
            factor = int(self.params.get("subsample_factor", 8))
            if (
                self.x_prev_subsampled is None
                or self.output_prev_norm is None
                or self.relative_transformation_rate is None
            ):
                return False
            current = _subsample(x.detach(), factor).float()
            previous = self.x_prev_subsampled.float().to(current.device)
            input_change = (current - previous).abs().mean()
            approx = (
                self.relative_transformation_rate.to(input_change.device)
                * input_change
            ) / self.output_prev_norm.to(input_change.device).clamp_min(1e-6)
            self.cumulative_change_rate += float(approx.item())
            return self.cumulative_change_rate < float(
                self.params.get("reuse_threshold", 0.2)
            )

        raise ValueError(f"unknown cache method: {self.method}")

    def cached_outputs(
        self, video_x: torch.Tensor, audio_x: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.skipped_calls += 1
        self.skipped_in_a_row += 1
        self.events.append(
            {
                "call_index": self.total_calls - 1,
                "progress": round(self.progress, 6),
                "method": self.method,
            }
        )

        fixed_mode = (
            str(self.params.get("reuse_mode", "last"))
            if self.method == "fixedskip"
            else ""
        )

        if fixed_mode == "blend_ema_last" and self.ema_video_output is not None:
            alpha = float(self.params.get("blend_alpha", 0.5))
            alpha = min(max(alpha, 0.0), 1.0)
            video = (1.0 - alpha) * self.last_video_output + alpha * (
                self.ema_video_output.to(self.last_video_output.device)
            )
            audio = None
            if audio_x is not None and self.last_audio_output is not None:
                audio = self.last_audio_output
                if self.ema_audio_output is not None:
                    audio = (1.0 - alpha) * self.last_audio_output + alpha * (
                        self.ema_audio_output.to(self.last_audio_output.device)
                    )
            return (
                video.to(video_x.device, video_x.dtype),
                _to_device_dtype(audio, audio_x) if audio_x is not None else None,
            )

        if (
            (self.method == "delta" or fixed_mode == "delta")
            and self.previous_video_output is not None
        ):
            scale = float(self.params.get("delta_scale", 1.0))
            video_delta = self.last_video_output - self.previous_video_output.to(
                self.last_video_output.device
            )
            video = self.last_video_output + scale * video_delta
            audio = None
            if audio_x is not None and self.last_audio_output is not None:
                if self.previous_audio_output is not None:
                    audio_delta = self.last_audio_output - self.previous_audio_output.to(
                        self.last_audio_output.device
                    )
                    audio = self.last_audio_output + scale * audio_delta
                else:
                    audio = self.last_audio_output
            return (
                video.to(video_x.device, video_x.dtype),
                _to_device_dtype(audio, audio_x) if audio_x is not None else None,
            )

        if (self.method == "ema" or fixed_mode == "ema") and self.ema_video_output is not None:
            return (
                self.ema_video_output.to(video_x.device, video_x.dtype),
                _to_device_dtype(self.ema_audio_output, audio_x)
                if audio_x is not None
                else None,
            )

        if (
            self.method in {"teacache", "easycache", "lazycache"}
            or fixed_mode == "residual"
        ) and self.cache_video_diff is not None:
            video = video_x + self.cache_video_diff.to(video_x.device, video_x.dtype)
            audio = None
            if audio_x is not None and self.cache_audio_diff is not None:
                audio = audio_x + self.cache_audio_diff.to(audio_x.device, audio_x.dtype)
            elif audio_x is not None and self.last_audio_output is not None:
                audio = self.last_audio_output.to(audio_x.device, audio_x.dtype)
            return video, audio

        return (
            self.last_video_output.to(video_x.device, video_x.dtype),
            _to_device_dtype(self.last_audio_output, audio_x)
            if audio_x is not None
            else None,
        )

    def update(
        self,
        video_x: torch.Tensor,
        audio_x: torch.Tensor | None,
        video_output: torch.Tensor | None,
        audio_output: torch.Tensor | None,
    ) -> None:
        self.real_calls += 1
        if video_output is None:
            self.skipped_in_a_row = 0
            return

        cache_device = str(self.params.get("cache_device", "default"))
        factor = int(self.params.get("subsample_factor", 8))
        previous_output = self.last_video_output
        self.previous_video_output = self.last_video_output
        self.previous_audio_output = self.last_audio_output
        self.last_video_output = _clone_tensor(video_output, device=cache_device)
        self.last_audio_output = _clone_tensor(audio_output, device=cache_device)
        self.last_x = _clone_tensor(video_x, device=cache_device)

        if self.ema_video_output is None:
            self.ema_video_output = _clone_tensor(video_output, device=cache_device)
            self.ema_audio_output = _clone_tensor(audio_output, device=cache_device)
        else:
            decay = float(self.params.get("ema_decay", 0.5))
            self.ema_video_output = _clone_tensor(
                decay * self.ema_video_output.to(video_output.device)
                + (1.0 - decay) * video_output.detach(),
                device=cache_device,
            )
            if audio_output is not None and self.ema_audio_output is not None:
                self.ema_audio_output = _clone_tensor(
                    decay * self.ema_audio_output.to(audio_output.device)
                    + (1.0 - decay) * audio_output.detach(),
                    device=cache_device,
                )

        if self.method in {"teacache", "easycache", "lazycache", "fixedskip"}:
            self.cache_video_diff = _clone_tensor(
                video_output.detach() - video_x.detach(), device=cache_device
            )
            if audio_x is not None and audio_output is not None:
                self.cache_audio_diff = _clone_tensor(
                    audio_output.detach() - audio_x.detach(), device=cache_device
                )
            if self.method in {"easycache", "lazycache"}:
                current_x_subsampled = _clone_tensor(
                    _subsample(video_x, factor), device=cache_device
                )
                current_output_subsampled = _clone_tensor(
                    _subsample(video_output, factor), device=cache_device
                )
                if (
                    self.output_prev_subsampled is not None
                    and previous_output is not None
                    and self.x_prev_subsampled is not None
                ):
                    out_prev = self.output_prev_subsampled.to(video_output.device)
                    out_change = (
                        _subsample(video_output, factor).float() - out_prev.float()
                    ).abs().mean()
                    x_prev = self.x_prev_subsampled.to(video_x.device)
                    input_change = (
                        _subsample(video_x, factor).float() - x_prev.float()
                    ).abs().mean()
                    if float(input_change.item()) > 1e-12:
                        self.relative_transformation_rate = _clone_tensor(
                            out_change / input_change, device=cache_device
                        )
                self.x_prev_subsampled = current_x_subsampled
                self.output_prev_subsampled = current_output_subsampled
                self.output_prev_norm = _clone_tensor(
                    _mean_abs(video_output), device=cache_device
                )
                self.cumulative_change_rate = 0.0
            if self.method == "teacache":
                self.accumulated_rel_l1_distance = 0.0

        self.skipped_in_a_row = 0


def make_ltx2_stage1_cache_core_from_env() -> LTX2Stage1CacheCoreController | None:
    if not _env_flag("SGLANG_LTX2_STAGE1_CACHE_CORE_ENABLED", False):
        return None

    preset_name = os.environ.get(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET", "12of15_delta05_29calls"
    )
    preset = _PRESETS.get(preset_name)
    if preset is None:
        logger.warning(
            "Unknown SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET=%r; using 12of15_delta05_29calls",
            preset_name,
        )
        preset_name = "12of15_delta05_29calls"
        preset = _PRESETS[preset_name]

    method = os.environ.get("SGLANG_LTX2_STAGE1_CACHE_CORE_METHOD", preset["method"])
    expected_calls = _env_int(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_EXPECTED_CALLS",
        int(preset["expected_calls"]),
    )
    params = dict(preset["params"])
    params["skip_indices"] = _env_int_list(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_SKIP_INDICES",
        list(params.get("skip_indices", [])),
    )
    params["reuse_mode"] = os.environ.get(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_REUSE_MODE",
        str(params.get("reuse_mode", "last")),
    )
    params["cache_device"] = os.environ.get(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_CACHE_DEVICE",
        str(params.get("cache_device", "default")),
    )
    params["max_skip_steps"] = _env_int(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_MAX_SKIP_STEPS",
        int(params.get("max_skip_steps", 1)),
    )
    params["start_percent"] = _env_float(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_START_PERCENT",
        float(params.get("start_percent", 0.0)),
    )
    params["end_percent"] = _env_float(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_END_PERCENT",
        float(params.get("end_percent", 1.0)),
    )
    params["delta_scale"] = _env_float(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_DELTA_SCALE",
        float(params.get("delta_scale", 1.0)),
    )
    params["ema_decay"] = _env_float(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_EMA_DECAY",
        float(params.get("ema_decay", 0.5)),
    )
    params["blend_alpha"] = _env_float(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_BLEND_ALPHA",
        float(params.get("blend_alpha", 0.5)),
    )
    params["warmup_steps"] = _env_int(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_WARMUP_STEPS",
        int(params.get("warmup_steps", 1)),
    )
    params["interval"] = _env_int(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_INTERVAL",
        int(params.get("interval", 1)),
    )
    params["subsample_factor"] = _env_int(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_SUBSAMPLE_FACTOR",
        int(params.get("subsample_factor", 8)),
    )
    params["rel_l1_thresh"] = _env_float(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_REL_L1_THRESH",
        float(params.get("rel_l1_thresh", 0.4)),
    )
    params["similarity_threshold"] = _env_float(
        "SGLANG_LTX2_STAGE1_CACHE_CORE_SIMILARITY_THRESHOLD",
        float(params.get("similarity_threshold", 0.05)),
    )

    controller = LTX2Stage1CacheCoreController(
        method, params, expected_calls=expected_calls
    )
    logger.info(
        "Installed LTX2 stage1 cache core preset=%s method=%s expected_calls=%s params=%s",
        preset_name,
        method,
        expected_calls,
        params,
    )
    return controller


__all__ = [
    "LTX2Stage1CacheCoreController",
    "make_ltx2_stage1_cache_core_from_env",
]
