# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for Hunyuan3D generation."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class Hunyuan3DSamplingParams(SamplingParams):
    """Sampling parameters for Hunyuan3D image-to-mesh generation."""

    negative_prompt: str = ""

    shape_num_inference_steps: int = 50
    guidance_scale: float = 5.0

    paint_num_inference_steps: int = 30
    paint_guidance_scale: float = 2.0

    def __post_init__(self):
        if self.prompt is None:
            self.prompt = ""

        if self.num_inference_steps is None:
            self.num_inference_steps = self.shape_num_inference_steps

        self.guidance_scale = max(5.0, min(self.guidance_scale, 6.5))
        super().__post_init__()
