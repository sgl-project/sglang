## Here's my code:


## here are the updates to merge:
# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class LongCatVideoSamplingParams(SamplingParams):
    # Basic t2v generation parameters (480p)
    num_inference_steps: int = 50
    num_frames: int = 93
    height: int = 480
    width: int = 832
    fps: int = 15
    guidance_scale: float = 4.0
