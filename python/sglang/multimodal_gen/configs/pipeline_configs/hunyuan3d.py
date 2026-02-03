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
    shape_mc_algo: Optional[str] = "mc"
    shape_num_chunks: int = 8000
    shape_output_type: str = "trimesh"

    # Delight model configuration
    delight_enable: bool = True
    delight_prompt: str = "3D image"
    delight_negative_prompt: str = "2D image"
    delight_strength: float = 0.8
    delight_num_inference_steps: int = 40
    delight_guidance_scale: float = 7.5
    delight_cfg_image: float = 1.5

    # Paint model configuration
    paint_enable: bool = False
    paint_num_inference_steps: int = 30
    paint_guidance_scale: float = 3.0
    paint_resolution: int = 512
    paint_render_size: int = 2048
    paint_texture_size: int = 2048
    paint_use_remesh: bool = True
    paint_save_glb: bool = True
    paint_turbo_mode: bool = False  # Enable turbo mode for faster inference
