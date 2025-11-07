# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.base import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


@dataclass
class HunyuanSamplingParams(SamplingParams):
    num_inference_steps: int = 50

    num_frames: int = 125
    height: int = 720
    width: int = 1280
    fps: int = 24

    guidance_scale: float = 1.0

    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.15,
            coefficients=[
                7.33226126e02,
                -4.01131952e02,
                6.75869174e01,
                -3.14987800e00,
                9.61237896e-02,
            ],
        )
    )


@dataclass
class FastHunyuanSamplingParam(HunyuanSamplingParams):
    num_inference_steps: int = 6
