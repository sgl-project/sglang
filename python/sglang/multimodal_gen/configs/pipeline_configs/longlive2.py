# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/NVlabs/LongLive

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig
from sglang.multimodal_gen.configs.models.dits.longlive2 import LongLive2VideoConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_TI2V_5B_Config
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class LongLive2T2VConfig(Wan2_2_TI2V_5B_Config):

    is_causal: bool = True
    task_type: ModelTaskType = ModelTaskType.TI2V
    vae_precision: str = "bf16"

    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    expand_timesteps: bool = False

    dit_config: DiTConfig = field(default_factory=LongLive2VideoConfig)
    def adjust_num_frames(self, num_frames: int) -> int:
        num_frames = super().adjust_num_frames(num_frames)
        vae_scale_factor_temporal = self.vae_config.arch_config.scale_factor_temporal
        latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        block_size = self.dit_config.arch_config.num_frames_per_block
        if latent_frames % block_size == 0:
            return num_frames

        adjusted_latent_frames = max(
            block_size, latent_frames // block_size * block_size
        )
        adjusted_num_frames = (
            adjusted_latent_frames - 1
        ) * vae_scale_factor_temporal + 1
        logger.warning(
            "`num_frames` must map to latent frames divisible by %s for "
            "LongLive2 causal denoising. Rounding from %s to %s.",
            block_size,
            num_frames,
            adjusted_num_frames,
        )
        return adjusted_num_frames

    def postprocess_image_latent(self, latent_condition, batch):
        return latent_condition[:, :, :1]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
