# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class MovaSamplingParams(SamplingParams):
    # Video parameters (MoVA defaults)
    height: int = 352
    width: int = 640
    num_frames: int = 193
    fps: int = 24

    # Denoising stage
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    sigma_shift: float = 5.0
    visual_shift: float = 5.0
    audio_shift: float = 5.0

    negative_prompt: str = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
    )
