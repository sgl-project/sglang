# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class GlmImageSamplingParams(SamplingParams):
    num_frames: int = 1
    guidance_scale: float = 1.5
    num_inference_steps: int = 30
