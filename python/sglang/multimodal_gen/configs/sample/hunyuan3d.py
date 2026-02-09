# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for Hunyuan3D generation."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class Hunyuan3DSamplingParams(SamplingParams):
    """Sampling parameters for Hunyuan3D image-to-mesh generation."""

    # Shape generation parameters
    shape_num_inference_steps: int = 50
    shape_guidance_scale: float = 5.0

    # Paint/texture generation parameters
    paint_num_inference_steps: int = 30
    paint_guidance_scale: float = 2.0
    seed: int = 0
    negative_prompt: str = ""

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Set default empty prompt for image-to-3D generation (prompt not required)
        if self.prompt is None:
            self.prompt = ""

        super().__post_init__()

        if self.shape_guidance_scale < 0:
            raise ValueError("shape_guidance_scale must be non-negative")

        if self.paint_num_inference_steps <= 0:
            raise ValueError("paint_num_inference_steps must be positive")

        if self.paint_guidance_scale < 0:
            raise ValueError("paint_guidance_scale must be non-negative")
