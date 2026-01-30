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

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Set default empty prompt for image-to-3D generation (prompt not required)
        if self.prompt is None:
            self.prompt = ""

        super().__post_init__()

        if self.shape_guidance_scale < 0:
            raise ValueError("shape_guidance_scale must be non-negative")
