# SPDX-License-Identifier: Apache-2.0
"""StableDiffusion3 sampling parameters configuration."""

import dataclasses

from sglang.multimodal_gen.configs.sample.base import SamplingParams


@dataclasses.dataclass
class StableDiffusion3SamplingParams(SamplingParams):
    """Sampling parameters for StableDiffusion3."""

    # SD3 specific parameters
    negative_prompt: str = " "
    num_frames: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.0
    guidance_rescale: float = 0.0
