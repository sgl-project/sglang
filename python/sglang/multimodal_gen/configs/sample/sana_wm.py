# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for SANA-WM TI2V world model generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.utils.sana_wm_camera import (
    SANA_WM_DEFAULT_PITCH_LIMIT_DEG,
    SANA_WM_DEFAULT_ROTATION_SPEED_DEG,
    SANA_WM_DEFAULT_TRANSLATION_SPEED,
    validate_sana_wm_motion_params,
)

# Type alias for camera tensor inputs accepted by SANA-WM.
# Downstream code in the stage coerces any of these to torch.Tensor.
CameraTensorLike = Union[
    torch.Tensor,
    np.ndarray,
    Sequence[Sequence[Sequence[float]]],
]


@dataclass
class SanaWMSamplingParams(SamplingParams):
    data_type: DataType = DataType.VIDEO

    # Resolution: 720p landscape (LTX-2 VAE requires multiples of 32).
    height: int = 704
    width: int = 1280

    # Frame count: 49 = (49-1)/8 = 6 latent frames → ~3 seconds at 16 fps.
    # NOTE: NVlabs' official script uses 161–321 frames for production quality.
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

    # --- Camera trajectory (6-DoF) — optional ---
    camera_to_world: Optional[CameraTensorLike] = None
    intrinsics: Optional[CameraTensorLike] = None
    camera_conditions: Optional[CameraTensorLike] = None
    chunk_plucker: Optional[CameraTensorLike] = None
    action: Optional[str] = None
    translation_speed: float = SANA_WM_DEFAULT_TRANSLATION_SPEED
    rotation_speed_deg: float = SANA_WM_DEFAULT_ROTATION_SPEED_DEG
    pitch_limit_deg: float = SANA_WM_DEFAULT_PITCH_LIMIT_DEG

    # Refiner control — use `skip_refiner` as the canonical field.
    skip_refiner: Optional[Union[bool, str, int]] = None
    refiner_prompt: Optional[Union[str, list[str]]] = None

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        for name in ("skip_refiner", "refiner_prompt"):
            value = getattr(self, name)
            if value is not None:
                extra[name] = value
        return extra

    def _adjust(self, server_args):
        super()._adjust(server_args)

        # Validate camera motion parameters.
        (
            self.translation_speed,
            self.rotation_speed_deg,
            self.pitch_limit_deg,
        ) = validate_sana_wm_motion_params(
            translation_speed=self.translation_speed,
            rotation_speed_deg=self.rotation_speed_deg,
            pitch_limit_deg=self.pitch_limit_deg,
        )

        # Mutual exclusion: only one camera specification allowed.
        if self.action is not None and self.camera_to_world is not None:
            raise ValueError(
                "SANA-WM accepts either `action` or `camera_to_world`, not both."
            )

        if self.camera_conditions is not None:
            self.condition_inputs["camera_conditions"] = self.camera_conditions
        if self.chunk_plucker is not None:
            self.condition_inputs["chunk_plucker"] = self.chunk_plucker
        if self.camera_to_world is not None:
            self.condition_inputs["camera_to_world"] = self.camera_to_world
        if self.intrinsics is not None:
            self.condition_inputs["intrinsics"] = self.intrinsics
        if self.action is not None:
            self.condition_inputs["action"] = self.action
            self.condition_inputs["translation_speed"] = self.translation_speed
            self.condition_inputs["rotation_speed_deg"] = self.rotation_speed_deg
            self.condition_inputs["pitch_limit_deg"] = self.pitch_limit_deg
