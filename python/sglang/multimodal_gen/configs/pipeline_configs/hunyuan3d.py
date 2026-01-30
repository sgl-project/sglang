# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


@dataclass
class Hunyuan3D2PipelineConfig(PipelineConfig):
    """Pipeline configuration for Hunyuan3D image-to-mesh generation."""

    shape_subfolder: str = "hunyuan3d-dit-v2-0"
    paint_subfolder: str = "hunyuan3d-paint-v2-0"
    delight_subfolder: str = "hunyuan3d-delight-v2-0"

    task_type: ModelTaskType = ModelTaskType.I2M

    # Shape model configuration
    shape_model_path: Optional[str] = None
    shape_use_safetensors: bool = True
    shape_variant: Optional[str] = "fp16"
    shape_num_inference_steps: int = 50
    shape_guidance_scale: float = 5.0
    shape_box_v: float = 1.01
    shape_octree_resolution: int = 384
    shape_mc_level: float = 0.0
    shape_mc_algo: Optional[str] = None
    shape_num_chunks: int = 8000
    shape_output_type: str = "trimesh"

    # Paint model configuration
    paint_enable: bool = False
    paint_num_inference_steps: int = 15
    paint_guidance_scale: float = 3.0
    paint_resolution: int = 512
    paint_texture_size: int = 4096
    paint_use_remesh: bool = True
    paint_save_glb: bool = True
    paint_turbo_mode: bool = False  # Enable turbo mode for faster inference
