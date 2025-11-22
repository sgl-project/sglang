# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.flux import FluxConfig
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfig,
    T5Config,
)
from sglang.multimodal_gen.configs.models.vaes.flux import FluxVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
    preprocess_text,
    shard_rotary_emb_for_sp,
)
from sglang.multimodal_gen.configs.pipeline_configs.hunyuan import (
    clip_postprocess_text,
    clip_preprocess_text,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import _pack_latents


def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    return outputs.last_hidden_state


@dataclass
class FluxPipelineConfig(ImagePipelineConfig):
    """Configuration for the FLUX pipeline."""

    embedded_cfg_scale: float = 3.5

    task_type: ModelTaskType = ModelTaskType.T2I

    vae_tiling: bool = False

    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=FluxConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=FluxVAEConfig)

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (CLIPTextConfig(), T5Config())
    )

    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", "bf16")
    )

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (clip_preprocess_text, preprocess_text),
    )

    postprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (clip_postprocess_text, t5_postprocess_text)
    )

    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            dict(
                max_length=77,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
            ),
            None,
        ]
    )

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        shape = (batch_size, num_channels_latents, height, width)
        return shape

    def maybe_pack_latents(self, latents, batch_size, batch):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        # pack latents
        return _pack_latents(latents, batch_size, num_channels_latents, height, width)

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[1]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[1]

    def _prepare_latent_image_ids(self, original_height, original_width, device):
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        height = int(original_height) // (vae_scale_factor * 2)
        width = int(original_width) // (vae_scale_factor * 2)
        latent_image_ids = torch.zeros(height, width, 3, device=device)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height, device=device)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width, device=device)[None, :]
        )

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids

    def get_freqs_cis(self, prompt_embeds, width, height, device, rotary_emb):
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device)
        img_ids = self._prepare_latent_image_ids(
            original_height=height,
            original_width=width,
            device=device,
        )

        # NOTE(mick): prepare it here, to avoid unnecessary computations
        img_cos, img_sin = rotary_emb.forward(img_ids)
        img_cos = shard_rotary_emb_for_sp(img_cos)
        img_sin = shard_rotary_emb_for_sp(img_sin)

        txt_cos, txt_sin = rotary_emb.forward(txt_ids)

        cos = torch.cat([txt_cos, img_cos], dim=0).to(device=device)
        sin = torch.cat([txt_sin, img_sin], dim=0).to(device=device)
        return cos, sin

    def post_denoising_loop(self, latents, batch):
        # unpack latents for flux
        (
            latents,
            batch_size,
            channels,
            height,
            width,
        ) = self._unpad_and_unpack_latents(latents, batch)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "freqs_cis": self.get_freqs_cis(
                batch.prompt_embeds[1], batch.width, batch.height, device, rotary_emb
            ),
            "pooled_projections": (
                batch.pooled_embeds[0] if batch.pooled_embeds else None
            ),
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "freqs_cis": self.get_freqs_cis(
                batch.negative_prompt_embeds[1],
                batch.width,
                batch.height,
                device,
                rotary_emb,
            ),
            "pooled_projections": (
                batch.neg_pooled_embeds[0] if batch.neg_pooled_embeds else None
            ),
        }
