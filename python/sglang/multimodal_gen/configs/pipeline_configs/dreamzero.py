# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import DreamZeroCausalWanConfig
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


def _dreamzero_text_encoder_config() -> EncoderConfig:
    return EncoderConfig(prefix="dreamzero_text_encoder")


def _dreamzero_image_encoder_config() -> EncoderConfig:
    return EncoderConfig(prefix="dreamzero_image_encoder")


@dataclass
class DreamZeroPipelineConfig(PipelineConfig):
    """Configuration for the DreamZero DROID one-shot action pipeline."""

    task_type: ModelTaskType = ModelTaskType.VLA_ACTION

    dit_config: DiTConfig = field(default_factory=DreamZeroCausalWanConfig)
    dit_precision: str = "bf16"

    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_precision: str = "bf16"
    vae_tiling: bool = False
    vae_sp: bool = False

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (_dreamzero_text_encoder_config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_extra_args: list[dict] = field(default_factory=lambda: [{}])

    image_encoder_config: EncoderConfig = field(
        default_factory=_dreamzero_image_encoder_config
    )
    image_encoder_precision: str = "bf16"
    dreamzero_compile_components: bool = True
    dreamzero_use_tensor_parallel: bool = False
    dreamzero_sequence_parallel_size: int = 1
    dreamzero_max_sessions: int = 10

    flow_shift: float | None = 5.0
    should_use_guidance: bool = True
    cfg_scale: float = 5.0

    embodiment_tag: str = "oxe_droid"
    action_horizon: int = 24
    num_inference_steps: int = 16
    dit_step_mask: tuple[bool, ...] | None = (
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        True,
        True,
    )
    dynamic_cache_schedule: bool = False
    relative_action_per_horizon: bool = False

    num_frames: int = 33
    synthetic_height: int = 180
    synthetic_width: int = 320
    latent_height: int = 44
    latent_width: int = 80

    tile_size_height: int = 34
    tile_size_width: int = 34
    tile_stride_height: int = 18
    tile_stride_width: int = 16
    tiled: bool = False

    def __post_init__(self) -> None:
        if self.dreamzero_sequence_parallel_size < 1:
            raise ValueError("dreamzero_sequence_parallel_size must be at least 1")
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = False
        self.vae_config.use_tiling = self.tiled
        self.vae_config.use_temporal_tiling = False
        self.vae_config.use_parallel_tiling = False
        self.vae_config.use_parallel_encode = False
        self.vae_config.use_parallel_decode = False
        self.vae_config.tile_sample_min_height = self.tile_size_height
        self.vae_config.tile_sample_min_width = self.tile_size_width
        self.vae_config.tile_sample_stride_height = self.tile_stride_height
        self.vae_config.tile_sample_stride_width = self.tile_stride_width

    def allow_set_num_frames(self):
        return False

    def adjust_num_frames(self, num_frames):
        return self.num_frames

    def supports_dynamic_batching(self):
        return False

    def estimate_request_cost(self, batch) -> float:
        return float(self.action_horizon)
