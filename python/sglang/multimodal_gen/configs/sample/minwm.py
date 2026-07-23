# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.wan import Wan2_2_TI2V_5B_SamplingParam


@dataclass
class MinWMSamplingParams(Wan2_2_TI2V_5B_SamplingParam):
    height: int = 480
    width: int = 832
    num_frames: int = 1
    fps: int = 24
    guidance_scale: float = 0.0
    num_inference_steps: int = 4
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [(832, 480), (480, 832)]
    )
