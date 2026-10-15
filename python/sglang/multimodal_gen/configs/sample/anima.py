# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class AnimaSamplingParams(SamplingParams):
    height: int = 1024
    width: int = 1024
    num_frames: int = 1
    num_inference_steps: int = 30
    guidance_scale: float = 4.0
    negative_prompt: str = ""
