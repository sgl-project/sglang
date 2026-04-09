# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Optional

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.hunyuan3d import Hunyuan3DDiTConfig
from sglang.multimodal_gen.configs.models.vaes.hunyuan3d import Hunyuan3DVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


@dataclass
class Hunyuan3D2PipelineConfig(PipelineConfig):
    """Pipeline configuration for Hunyuan3D image-to-mesh generation."""

    task_type: ModelTaskType = ModelTaskType.I2M

    # Subfolder paths
    shape_subfolder: str = "hunyuan3d-dit-v2-0"
    paint_subfolder: str = "hunyuan3d-paint-v2-0"
    delight_subfolder: str = "hunyuan3d-delight-v2-0"

    # DiT configuration
    dit_config: DiTConfig = field(default_factory=Hunyuan3DDiTConfig)
    dit_precision: str = "fp16"

    # VAE configuration
    vae_config: VAEConfig = field(default_factory=Hunyuan3DVAEConfig)
    vae_precision: str = "fp32"

    # Shape model configuration
    shape_model_path: Optional[str] = None
    shape_use_safetensors: bool = True
    shape_variant: Optional[str] = "fp16"
    shape_num_inference_steps: int = 50
    guidance_scale: float = 5.0
    shape_box_v: float = 1.01
    shape_octree_resolution: int = 384
    shape_mc_level: float = 0.0
    shape_mc_algo: Optional[str] = "mc"
    shape_num_chunks: int = 8000
    shape_output_type: str = "trimesh"

    # Delight model configuration
    delight_enable: bool = True
    delight_prompt: str = ""
    delight_negative_prompt: str = ""
    delight_strength: float = 1.0
    delight_num_inference_steps: int = 50
    delight_guidance_scale: float = 1.0
    delight_cfg_image: float = 1.5

    # Paint model configuration
    paint_enable: bool = True
    paint_num_inference_steps: int = 30
    paint_guidance_scale: float = 2.0
    paint_resolution: int = 512
    paint_render_size: int = 2048
    paint_texture_size: int = 2048
    paint_use_remesh: bool = True
    paint_save_glb: bool = True
    paint_turbo_mode: bool = False

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        latent_shape = self.vae_config.arch_config.latent_shape
        shape = (batch_size, *latent_shape)
        return shape
