# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.base import SamplingParams


@dataclass
class FluxSamplingParams(SamplingParams):
    # Video parameters
    # height: int = 1024
    # width: int = 1024
    num_frames: int = 1
    # Denoising stage
    guidance_scale: float = 1.0
    negative_prompt: str = None
    num_inference_steps: int = 50
