# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class FluxSamplingParams(SamplingParams):
    _default_height: ClassVar[int] = 128 * 8  # default_sample_size * vae_scale_factor
    _default_width: ClassVar[int] = 128 * 8

    num_frames: int = 1
    # Denoising stage
    guidance_scale: float = 3.5
    negative_prompt: str = None
    num_inference_steps: int = 50


@dataclass
class Flux2SamplingParams(FluxSamplingParams):
    guidance_scale: float = 4.0


@dataclass
class Flux2KleinSamplingParams(Flux2SamplingParams):
    # Klein is step-distilled, so default to 4 steps
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
