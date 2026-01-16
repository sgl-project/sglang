# SPDX-License-Identifier: Apache-2.0
"""
MoVA pipeline configuration.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.models.dits import MovaAudioConfig, MovaVideoConfig
from sglang.multimodal_gen.configs.models.encoders import T5Config
from sglang.multimodal_gen.configs.models.vaes import DacVAEConfig, WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import t5_postprocess_text
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class MovaPipelineConfig(PipelineConfig):
    """Configuration for MoVA (text+image -> video+audio) pipelines."""

    task_type: ModelTaskType = ModelTaskType.TI2V

    # Model configs
    video_dit_config: MovaVideoConfig = field(default_factory=MovaVideoConfig)
    video_dit2_config: MovaVideoConfig = field(default_factory=MovaVideoConfig)
    audio_dit_config: MovaAudioConfig = field(default_factory=MovaAudioConfig)

    # Video VAE (Wan) + Audio VAE (DAC)
    vae_config: WanVAEConfig = field(default_factory=WanVAEConfig)
    audio_vae_config: DacVAEConfig = field(default_factory=DacVAEConfig)

    # Text encoder (UMT5 compatible)
    text_encoder_configs: tuple = field(default_factory=lambda: (T5Config(),))
    postprocess_text_funcs: tuple = field(
        default_factory=lambda: (t5_postprocess_text,)
    )

    # MoVA specific
    audio_vae_type: str = "dac"
    boundary_ratio: float | None = 0.9

    # temporal alignment: MoVA expects (num_frames - 1) % 4 == 0
    time_division_factor: int = 4
    time_division_remainder: int = 1

    def _center_crop_and_resize(
        self, image: Image.Image, target_height: int, target_width: int
    ) -> Image.Image:
        image_np = np.array(image)
        image_height, image_width, _ = image_np.shape
        if image_height / image_width < target_height / target_width:
            cropped_width = int(image_height / target_height * target_width)
            left = (image_width - cropped_width) // 2
            image_np = image_np[:, left : left + cropped_width]
            return Image.fromarray(image_np).resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )
        cropped_height = int(image_width / target_width * target_height)
        top = (image_height - cropped_height) // 2
        image_np = image_np[top : top + cropped_height, :]
        return Image.fromarray(image_np).resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )

    def adjust_num_frames(self, num_frames: int) -> int:
        if num_frames is None:
            return num_frames
        if num_frames % self.time_division_factor != self.time_division_remainder:
            adjusted = (
                (num_frames + self.time_division_factor - 1)
                // self.time_division_factor
                * self.time_division_factor
                + self.time_division_remainder
            )
            logger.warning(
                "`num_frames` (%s) is not compatible with MoVA temporal constraints. "
                "Rounding to %s.",
                num_frames,
                adjusted,
            )
            return adjusted
        return num_frames

    def preprocess_condition_image(
        self, image, target_width, target_height, _vae_image_processor
    ):
        image = self._center_crop_and_resize(image, target_height, target_width)
        return image, (target_width, target_height)

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        spatial = self.vae_config.arch_config.spatial_compression_ratio
        length = (num_frames - 1) // self.time_division_factor + 1
        shape = (
            batch_size,
            self.video_dit_config.arch_config.out_dim,
            length,
            batch.height // spatial,
            batch.width // spatial,
        )
        return shape

    def prepare_audio_latent_shape(self, batch_size, num_samples, audio_vae):
        if self.audio_vae_type == "oobleck":
            latent_T = num_samples // audio_vae.hop_length
        else:
            latent_T = (num_samples + audio_vae.hop_length - 1) // audio_vae.hop_length
        return (batch_size, audio_vae.latent_dim, latent_T)

    def normalize_video_latents(self, latents: torch.Tensor, video_vae) -> torch.Tensor:
        latents_mean = getattr(video_vae.config, "latents_mean", None)
        latents_std = getattr(video_vae.config, "latents_std", None)
        if latents_mean is None or latents_std is None:
            return latents
        mean = torch.tensor(
            latents_mean, device=latents.device, dtype=latents.dtype
        ).view(1, video_vae.config.z_dim, 1, 1, 1)
        inv_std = (
            1.0 / torch.tensor(latents_std, device=latents.device, dtype=latents.dtype)
        ).view(1, video_vae.config.z_dim, 1, 1, 1)
        return (latents - mean) * inv_std

    def denormalize_video_latents(
        self, latents: torch.Tensor, video_vae
    ) -> torch.Tensor:
        latents_mean = getattr(video_vae.config, "latents_mean", None)
        latents_std = getattr(video_vae.config, "latents_std", None)
        if latents_mean is None or latents_std is None:
            return latents
        mean = torch.tensor(
            latents_mean, device=latents.device, dtype=latents.dtype
        ).view(1, video_vae.config.z_dim, 1, 1, 1)
        std = torch.tensor(
            latents_std, device=latents.device, dtype=latents.dtype
        ).view(1, video_vae.config.z_dim, 1, 1, 1)
        return latents * std + mean
