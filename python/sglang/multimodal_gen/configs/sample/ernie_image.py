# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for ErnieImage."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class ErnieImageSamplingParams(SamplingParams):
    negative_prompt: str = " "
    num_frames: int = 1
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    use_pe: bool = True
