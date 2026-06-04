# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.wan import Wan2_2_TI2V_5B_SamplingParam


@dataclass
class LongLive2SamplingParams(Wan2_2_TI2V_5B_SamplingParam):
    height: int = 704
    width: int = 1280
    fps: int = 24
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    num_frames: int = 61

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 704),
            (704, 1280),
            (832, 480),
            (480, 832),
        ]
    )
