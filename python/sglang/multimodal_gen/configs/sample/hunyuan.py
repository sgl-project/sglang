# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


@dataclass
class HunyuanSamplingParams(SamplingParams):
    num_inference_steps: int = 50

    num_frames: int = 125
    height: int = 720
    width: int = 1280
    fps: int = 24

    guidance_scale: float = 1.0

    # HunyuanVideo supported resolutions
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            # 540p resolutions
            (960, 544),  # 9:16
            (544, 960),  # 16:9
            (832, 624),  # 4:3
            (624, 832),  # 3:4
            (720, 720),  # 1:1
            # 720p resolutions (recommended)
            (1280, 720),  # 9:16
            (720, 1280),  # 16:9
            (832, 1104),  # 4:3
            (1104, 832),  # 3:4
            (960, 960),  # 1:1
        ]
    )

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
