# SPDX-License-Identifier: Apache-2.0
"""
MOVA pipeline configuration.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.multimodal_gen.configs.models.dits import MOVAAudioConfig, MOVAVideoConfig
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
class MOVAPipelineConfig(PipelineConfig):
    """Configuration for MOVA (text+image -> video+audio) pipelines."""

    task_type: ModelTaskType = ModelTaskType.T2V

    # Model configs
    dit_config: MOVAVideoConfig = field(default_factory=MOVAVideoConfig)
    audio_dit_config: MOVAAudioConfig = field(default_factory=MOVAAudioConfig)

    # Video VAE (Wan) + Audio VAE (DAC)
    vae_config: WanVAEConfig = field(default_factory=WanVAEConfig)
    audio_vae_config: DacVAEConfig = field(default_factory=DacVAEConfig)
    audio_vae_precision: str = "fp32"

    # Text encoder (UMT5 compatible)
    text_encoder_configs: tuple = field(default_factory=lambda: (T5Config(),))
    postprocess_text_funcs: tuple = field(
        default_factory=lambda: (t5_postprocess_text,)
    )

    # MOVA specific
    audio_vae_type: str = "dac"
    boundary_ratio: float | None = 0.9

    # temporal alignment: MOVA expects (num_frames - 1) % 4 == 0
    time_division_factor: int = 4
    time_division_remainder: int = 1

    def _center_crop_and_resize(
        self, image: torch.Tensor | Image.Image, target_height: int, target_width: int
    ) -> torch.Tensor | Image.Image:
        if not isinstance(image, (Image.Image, torch.Tensor)):
            raise TypeError(f"Unsupported image type: {type(image)}")
        if isinstance(image, Image.Image):
            image = torch.from_numpy(np.array(image))

        if image.ndim == 2:
            image = image[..., None]

        if not image.dtype.is_floating_point:
            image = image.to(torch.float32).div(255.0)

        if image.ndim == 3:
            if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
                image = image.unsqueeze(0)
            else:
                image = image.permute(2, 0, 1).unsqueeze(0)
        elif image.ndim == 4:
            if image.shape[1] not in (1, 3, 4) and image.shape[-1] in (1, 3, 4):
                image = image.permute(0, 3, 1, 2)

        image_height, image_width = image.shape[-2], image.shape[-1]
        if image_height == target_height and image_width == target_width:
            return image

        logger.info(
            "Center cropping and resizing image to %dx%d", target_width, target_height
        )

        if image_height * target_width < image_width * target_height:
            cropped_width = (image_height * target_width) // target_height
            left = (image_width - cropped_width) // 2
            image = image[..., :, left : left + cropped_width]
        else:
            cropped_height = (image_width * target_height) // target_width
            top = (image_height - cropped_height) // 2
            image = image[..., top : top + cropped_height, :]

        image = F.interpolate(
            image,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return image

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
                "`num_frames` (%s) is not compatible with MOVA temporal constraints. "
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
            self.dit_config.arch_config.out_dim,
            length,
            batch.height // spatial,
            batch.width // spatial,
        )
        return shape

    def prepare_audio_latent_shape(self, batch_size, num_samples, audio_vae):
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


@dataclass
class MOVA360PConfig(MOVAPipelineConfig):
    """Configuration for MOVA 360P (text+image -> video+audio) pipelines."""

    max_area: int = 352 * 640


@dataclass
class MOVA720PConfig(MOVAPipelineConfig):
    """Configuration for MOVA 720P (text+image -> video+audio) pipelines."""

    max_area: int = 720 * 1280
