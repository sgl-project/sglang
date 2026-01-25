# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class MOVASamplingParams(SamplingParams):
    # Video parameters (MOVA defaults)
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
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )


@dataclass
class MOVA_360P_SamplingParams(MOVASamplingParams):
    # Video parameters (MOVA 360P)
    height: int = 352
    width: int = 640

    # MOVA 360P supported resolutions
    supported_resolutions: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (352, 640),
            (640, 352),
        ]
    )


@dataclass
class MOVA_720P_SamplingParams(MOVASamplingParams):
    # Video parameters (MOVA 720P)
    height: int = 720
    width: int = 1280

    # MOVA 720P supported resolutions
    supported_resolutions: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (720, 1280),
            (1280, 720),
        ]
    )
