# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


@dataclass
class Hunyuan3DPipelineConfig(PipelineConfig):
    """Pipeline configuration for Hunyuan3D image-to-mesh generation."""

    task_type: ModelTaskType = ModelTaskType.I2M

    # Local code repo path (for hy3dshape/hy3dpaint imports).
    hunyuan3d_repo_path: str = (
        "python/sglang/multimodal_gen/runtime/models/hunyuan3d"
    )

    # Shape model configuration
    shape_model_path: str | None = None
    shape_subfolder: str = "hunyuan3d-dit-v2-1"
    shape_use_safetensors: bool = False
    shape_variant: str | None = "fp16"
    shape_num_inference_steps: int = 50
    shape_guidance_scale: float = 5.0
    shape_box_v: float = 1.01
    shape_octree_resolution: int = 384
    shape_mc_level: float = 0.0
    shape_mc_algo: str | None = None
    shape_num_chunks: int = 8000
    shape_output_type: str = "trimesh"

    # Paint model configuration
    paint_enable: bool = False
    paint_model_path: str | None = None
    paint_max_num_view: int = 6
    paint_resolution: int = 512
    paint_use_remesh: bool = True
    paint_save_glb: bool = True
