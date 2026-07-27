# SPDX-License-Identifier: Apache-2.0
"""Sampling defaults for BAGEL text-to-image generation."""

from dataclasses import dataclass
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class BagelSamplingParams(SamplingParams):
    """Sampling parameters supported by the first BAGEL T2I integration."""

    _default_height: ClassVar[int] = 1024
    _default_width: ClassVar[int] = 1024

    num_frames: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    negative_prompt: str | None = None
    flow_shift: float | None = 3.0
