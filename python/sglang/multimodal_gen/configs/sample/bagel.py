# SPDX-License-Identifier: Apache-2.0
"""Sampling defaults for BAGEL generation, editing, and understanding."""

import math
from dataclasses import dataclass
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class BagelSamplingParams(SamplingParams):
    """Shared sampling parameters for BAGEL image generation."""

    _default_height: ClassVar[int] = 1024
    _default_width: ClassVar[int] = 1024

    num_frames: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    negative_prompt: str | None = None
    flow_shift: float | None = 3.0
    enable_taylorseer: bool = False

    def _validate(self) -> None:
        super()._validate()
        if not isinstance(self.enable_taylorseer, bool):
            raise ValueError("enable_taylorseer must be a boolean")


@dataclass
class BagelThinkingSamplingParams(BagelSamplingParams):
    """Sampling parameters for deterministic text planning before T2I."""

    max_think_tokens: int = 1000
    think_do_sample: bool = False
    think_temperature: float = 0.3

    def _validate(self) -> None:
        super()._validate()
        if (
            not isinstance(self.max_think_tokens, int)
            or isinstance(self.max_think_tokens, bool)
            or self.max_think_tokens <= 0
        ):
            raise ValueError("max_think_tokens must be a positive integer")
        if not isinstance(self.think_do_sample, bool):
            raise ValueError("think_do_sample must be a boolean")
        if self.think_do_sample and (
            not isinstance(self.think_temperature, (int, float))
            or isinstance(self.think_temperature, bool)
            or not math.isfinite(float(self.think_temperature))
            or self.think_temperature <= 0
        ):
            raise ValueError(
                "think_temperature must be finite and positive when sampling"
            )


@dataclass
class BagelUnderstandingSamplingParams(BagelSamplingParams):
    """Sampling parameters for image-conditioned autoregressive text."""

    num_inference_steps: int = 1
    guidance_scale: float = 1.0
    negative_prompt: str | None = None
    save_output: bool = False
    return_file_paths_only: bool = False

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.3
    enable_thinking: bool = False

    def _validate(self) -> None:
        super()._validate()
        if self.enable_taylorseer:
            raise ValueError(
                "enable_taylorseer is unavailable for BAGEL Understanding because "
                "it does not run image denoising"
            )
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(self.do_sample, bool):
            raise ValueError("do_sample must be a boolean")
        if not isinstance(self.enable_thinking, bool):
            raise ValueError("enable_thinking must be a boolean")
        if self.do_sample and (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not math.isfinite(float(self.temperature))
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive when sampling")


@dataclass
class BagelEditSamplingParams(BagelSamplingParams):
    """Sampling parameters for the explicit BAGEL Editing pipeline.

    ``guidance_scale`` controls text CFG. ``true_cfg_scale`` reuses the
    existing Images API field for BAGEL's image CFG and defaults to the
    pipeline's official Editing value when omitted.
    """

    true_cfg_scale: float | None = None
