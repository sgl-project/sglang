# SPDX-License-Identifier: Apache-2.0
import dataclasses
from dataclasses import field

from sglang.multimodal_gen.configs.pipeline_configs.joy_echo import (
    JOY_ECHO_DEFAULT_SIGMAS,
)
from sglang.multimodal_gen.configs.sample.ltx_2 import LTX2SamplingParams


@dataclasses.dataclass
class JoyEchoSamplingParams(LTX2SamplingParams):
    """Sampling parameters for JoyEcho DMD inference."""

    seed: int = 12345
    generator_device: str = "cuda"

    height: int = 480
    width: int = 832
    num_frames: int = 121
    fps: int = 25

    guidance_scale: float = 1.0
    num_inference_steps: int = 8

    sigmas: tuple[float, ...] = field(default_factory=lambda: JOY_ECHO_DEFAULT_SIGMAS)

    negative_prompt: str | None = None

    video_cfg_scale: float = 1.0
    audio_cfg_scale: float = 1.0

    enable_memory_bank: bool = True
    reset_memory_bank: bool = True
