# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class FluxSamplingParams(SamplingParams):
    num_frames: int = 1
    # Denoising stage
    guidance_scale: float = 1.0
    negative_prompt: str = None
    num_inference_steps: int = 50

    def __post_init__(self):
        default_sample_size = 128
        vae_scale_factor = 8
        # FIXME
        # self.height = default_sample_size * vae_scale_factor
        # self.width = default_sample_size * vae_scale_factor
