# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class BooguImageSamplingParams(SamplingParams):
    num_inference_steps: int = 50

    num_frames: int = 1
    height: int = 1024
    width: int = 1024

    guidance_scale: float = 4.0
    negative_prompt: str = ""
    max_sequence_length: int = 1280
