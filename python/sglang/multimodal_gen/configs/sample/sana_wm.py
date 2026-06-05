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
    """Default sampling parameters for SANA-WM 720p (704×1280) 16fps video.

    Frame counts must satisfy (num_frames - 1) % 8 == 0.

    Optional camera conditioning:
        camera_to_world: (T, 4, 4) extrinsics, one per output frame.
        intrinsics:      (T, 3, 3) pinhole intrinsics, one per output frame.
        action:          WASD/IJKL action DSL (e.g. "w-80,jw-40,w-40"), rolled
                         out to camera_to_world before the camera branch.
    Omitted camera_to_world -> static identity camera. Omitted intrinsics ->
    centered heuristic; pass explicit intrinsics for closest NVlabs parity.
    """

    data_type: DataType = DataType.VIDEO

    # Resolution: 720p landscape (LTX-2 VAE requires multiples of 32)
    height: int = 704
    width: int = 1280

    # 49 = (49-1)/8 = 6 latent frames → ~3 seconds at 16fps
    num_frames: int = 49

    # SANA-WM is trained at 16fps (override base default of 24).
    fps: int = 16

    num_inference_steps: int = 20

    guidance_scale: float = 4.5

    # NVlabs' SANA-WM inference defaults to an empty negative prompt.
    negative_prompt: str = ""

    # --- Camera trajectory (6-DoF) — optional ---
    camera_to_world: Optional[CameraTensorLike] = None
    intrinsics: Optional[CameraTensorLike] = None
    action: Optional[str] = None
    translation_speed: float = 0.04  # match official streaming (STREAMING_TRANSLATION_SPEED)
    rotation_speed_deg: float = 1.2
    pitch_limit_deg: float = 85.0

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        if self.action is not None and self.camera_to_world is not None:
            raise ValueError(
                "SANA-WM accepts either action or camera_to_world, not both."
            )
        if self.camera_to_world is not None:
            extra["camera_to_world"] = self.camera_to_world
        if self.intrinsics is not None:
            extra["intrinsics"] = self.intrinsics
        if self.action is not None:
            extra["action"] = self.action
            extra["translation_speed"] = self.translation_speed
            extra["rotation_speed_deg"] = self.rotation_speed_deg
            extra["pitch_limit_deg"] = self.pitch_limit_deg
        return extra
