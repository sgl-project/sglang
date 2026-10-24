# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.task_type import ModelTaskType


@dataclass
class OvisImageSamplingParams(SamplingParams):
    height: int = 1024
    width: int = 1024
    num_frames: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    negative_prompt: str = ""
    max_sequence_length: int = 256

    def _validate(self) -> None:
        super()._validate()
        for name in ("height", "width"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 16
                or value % 16 != 0
            ):
                raise ValueError(
                    f"Ovis-Image {name} must be an integer multiple of 16 and at least 16, got {value!r}"
                )
        if (
            self.task_type is not None
            and ModelTaskType.parse(self.task_type) != ModelTaskType.T2I
        ):
            raise ValueError("Ovis-Image supports text-to-image generation only")
        if self.image_path is not None or self.video_path is not None:
            raise ValueError("Ovis-Image does not support image or video conditioning")
        if self.num_frames != 1:
            raise ValueError("Ovis-Image requires num_frames=1")
        if self.max_sequence_length is not None and (
            isinstance(self.max_sequence_length, bool)
            or not isinstance(self.max_sequence_length, int)
            or not 1 <= self.max_sequence_length <= 256
        ):
            raise ValueError("Ovis-Image max_sequence_length must be between 1 and 256")
        for name in ("enable_teacache", "enable_spectrum", "enable_cache_dit"):
            if getattr(self, name):
                raise ValueError(f"Ovis-Image does not support {name}")
