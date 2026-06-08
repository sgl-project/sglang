# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for SANA-WM TI2V world model generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class SanaWMSamplingParams(SamplingParams):
    data_type: DataType = DataType.VIDEO

    # Resolution: 720p landscape (LTX-2 VAE requires multiples of 32).
    height: int = 704
    width: int = 1280

    # Frame count: 49 = (49-1)/8 = 6 latent frames, about 3 seconds at 16 fps.
    # NOTE: NVlabs' official script uses 161-321 frames for production quality.
    # 49 is a CI/speed-friendly default. For best quality use num_frames=161,
    # num_inference_steps=60, guidance_scale=5.0.
    num_frames: int = 49

    # SANA-WM is trained at 16 fps (override base default of 24).
    fps: int = 16

    # 20 steps is a CI-friendly default. NVlabs' official script uses 60 steps.
    num_inference_steps: int = 20

    # Classifier-free guidance scale. NVlabs' official default is 5.0.
    guidance_scale: float = 4.5

    # NVlabs' SANA-WM inference defaults to an empty negative prompt.
    negative_prompt: str = ""

    # --- Camera trajectory (6-DoF) - optional ---
    camera_to_world: Any | None = None
    intrinsics: Any | None = None
    camera_conditions: Any | None = None
    chunk_plucker: Any | None = None
    action: str | None = None
    translation_speed: float = 0.05
    rotation_speed_deg: float = 1.2
    pitch_limit_deg: float = 85.0

    # Refiner control; use `skip_refiner` as the canonical field.
    skip_refiner: bool | str | int | None = None
    refiner_prompt: str | list[str] | None = None
    refiner_seed: int | list[int] | None = None
    sink_size: int | None = None

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        for name in ("skip_refiner", "refiner_prompt", "refiner_seed", "sink_size"):
            value = getattr(self, name)
            if value is not None:
                extra[name] = value
        return extra

    def _adjust(self, server_args):
        super()._adjust(server_args)
        if self.condition_inputs is None:
            self.condition_inputs = {}

        for name in (
            "camera_conditions",
            "chunk_plucker",
            "camera_to_world",
            "intrinsics",
            "action",
        ):
            value = getattr(self, name)
            if value is not None:
                self.condition_inputs[name] = value

        if self.action is not None:
            self.condition_inputs["translation_speed"] = self.translation_speed
            self.condition_inputs["rotation_speed_deg"] = self.rotation_speed_deg
            self.condition_inputs["pitch_limit_deg"] = self.pitch_limit_deg

        if isinstance(self.sink_size, bool) or (
            self.sink_size is not None
            and (not isinstance(self.sink_size, int) or self.sink_size <= 0)
        ):
            raise ValueError(
                f"sink_size must be a positive int when set, got {self.sink_size!r}"
            )

        if isinstance(self.refiner_seed, list):
            if not self.refiner_seed:
                raise ValueError("refiner_seed list must not be empty")
            for seed in self.refiner_seed:
                if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                    raise ValueError(
                        "refiner_seed list must contain non-negative ints, "
                        f"got {self.refiner_seed!r}"
                    )
        elif self.refiner_seed is not None and (
            isinstance(self.refiner_seed, bool)
            or not isinstance(self.refiner_seed, int)
            or self.refiner_seed < 0
        ):
            raise ValueError(
                "refiner_seed must be a non-negative int, list of ints, or None, "
                f"got {self.refiner_seed!r}"
            )
