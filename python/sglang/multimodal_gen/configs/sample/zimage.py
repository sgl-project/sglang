# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


@dataclass
class ZImageTurboSamplingParams(SamplingParams):
    num_inference_steps: int = 9

    num_frames: int = 1
    negative_prompt: str = None
    # Z-Image officially recommends starting at 1024x1024 (it accepts any
    # /16-aligned resolution roughly in the 768-1280 grid, up to ~2MP).
    # Declaring the default lets server-based warmup specialize kernels for the
    # common request shape; supported_resolutions stays None ("all allowed") so
    # other resolutions still work without spurious "unsupported" warnings.
    height: int = 1024
    width: int = 1024
    # fps: int = 24

    guidance_scale: float = 0.0
    cfg_normalization: float | bool = False

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
class ZImageSamplingParams(SamplingParams):
    num_inference_steps: int = 50

    num_frames: int = 1
    negative_prompt: str = " "
    guidance_scale: float = 5.0
    cfg_normalization: float | bool = True
