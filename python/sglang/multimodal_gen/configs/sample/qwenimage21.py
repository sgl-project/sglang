# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import (
    MissingSidePolicy,
    SamplingParams,
)


@dataclass
class QwenImage21SamplingParams(SamplingParams):
    _default_height: ClassVar[int] = 1024
    _default_width: ClassVar[int] = 1024
    _request_size_model_name = "Qwen-Image 2.1"
    _request_size_missing_side = MissingSidePolicy.PER_SIDE
    # Align before request batching and pipeline validation consume the size.
    _request_size_aligned_at_init = True
    num_frames: int = 1
    guidance_scale: float = 1.0
    num_inference_steps: int = 40
    negative_prompt: str | None = None

    def _request_size_multiple(self, pipeline_config) -> int | None:
        # Matches Diffusers on every backend, so there is no pipeline_config guard.
        del pipeline_config
        return 32

    @classmethod
    def default_image_output_format(cls) -> str:
        """Preserve Qwen-Image 2.1's RGBA outputs by default."""

        return "png"
