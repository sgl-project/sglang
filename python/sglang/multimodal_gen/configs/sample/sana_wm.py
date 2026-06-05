# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for SANA-WM TI2V world model generation."""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)

# Type alias for the camera tensor inputs. Accept torch.Tensor, numpy arrays,
# or nested Python lists — coerced to torch.Tensor downstream in the stage.
CameraTensorLike = Union[Any, Sequence[Sequence[Sequence[float]]]]


@dataclass
class SanaWMSamplingParams(SamplingParams):
    data_type: DataType = DataType.VIDEO

    # Resolution: 720p landscape (LTX-2 VAE requires multiples of 32)
    height: int = 704
    width: int = 1280

    # Frame count: 49 = (49-1)/8 = 6 latent frames → ~3 seconds at 16fps
    num_frames: int = 49

    # SANA-WM is trained at 16fps (override base default of 24).
    fps: int = 16

    # SANA-WM inference steps: 20 steps is a good default for quality/speed
    num_inference_steps: int = 20

    # Classifier-free guidance scale
    guidance_scale: float = 4.5

    # NVlabs' SANA-WM inference defaults to an empty negative prompt.
    negative_prompt: str = ""

    # --- Camera trajectory (6-DoF) — optional ---
    camera_to_world: Optional[CameraTensorLike] = None
    intrinsics: Optional[CameraTensorLike] = None
    action: Optional[str] = None
    translation_speed: float = 0.05
    rotation_speed_deg: float = 1.2
    pitch_limit_deg: float = 85.0

    def _adjust(self, server_args):
        super()._adjust(server_args)
        if self.action is not None and self.camera_to_world is not None:
            raise ValueError(
                "SANA-WM accepts either action or camera_to_world, not both."
            )
        if self.camera_to_world is not None:
            self.condition_inputs["camera_to_world"] = self.camera_to_world
        if self.intrinsics is not None:
            self.condition_inputs["intrinsics"] = self.intrinsics
        if self.action is not None:
            self.condition_inputs["action"] = self.action
            self.condition_inputs["translation_speed"] = self.translation_speed
            self.condition_inputs["rotation_speed_deg"] = self.rotation_speed_deg
            self.condition_inputs["pitch_limit_deg"] = self.pitch_limit_deg
