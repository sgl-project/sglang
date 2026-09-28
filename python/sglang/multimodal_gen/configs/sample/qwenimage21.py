# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class QwenImage21SamplingParams(SamplingParams):
    _default_height: ClassVar[int] = 1024
    _default_width: ClassVar[int] = 1024
    num_frames: int = 1
    guidance_scale: float = 1.0
    num_inference_steps: int = 40
    negative_prompt: str | None = None
