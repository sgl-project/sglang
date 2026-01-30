# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for Hunyuan3D generation."""

from dataclasses import dataclass
from typing import Optional

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class Hunyuan3DSamplingParams(SamplingParams):
    """Sampling parameters for Hunyuan3D image-to-mesh generation."""

    # Shape generation parameters
    shape_num_inference_steps: int = 50
    shape_guidance_scale: float = 5.0
    shape_eta: float = 0.0

    # Mesh export parameters
    octree_resolution: int = 384
    mc_level: float = 0.0
    mc_algo: Optional[str] = None
    box_v: float = 1.01
    num_chunks: int = 8000

    # Output format
    output_type: str = "trimesh"  # "trimesh" | "latent"
    output_format: str = "obj"  # "obj" | "glb"

    # Paint parameters (when enabled)
    paint_enable: bool = False
    paint_num_inference_steps: int = 15
    paint_guidance_scale: float = 3.0
    paint_max_num_view: int = 6
    paint_resolution: int = 512
    paint_use_super_resolution: bool = True

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Set default empty prompt for image-to-3D generation (prompt not required)
        if self.prompt is None:
            self.prompt = ""

        super().__post_init__()

        if self.shape_guidance_scale < 0:
            raise ValueError("shape_guidance_scale must be non-negative")

        if self.octree_resolution not in [256, 384, 512]:
            # Allow but warn about non-standard resolutions
            pass

        if self.output_type not in ["trimesh", "latent"]:
            raise ValueError(f"Invalid output_type: {self.output_type}")

        if self.output_format not in ["obj", "glb"]:
            raise ValueError(f"Invalid output_format: {self.output_format}")
