from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.glmimage import GlmImageDitConfig
from sglang.multimodal_gen.configs.models.vaes.glmimage import GlmImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


@dataclass
class GlmImagePipelineConfig(ImagePipelineConfig):
    """Configuration for the GlmImage pipeline."""

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2I

    vae_tiling: bool = False

    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=GlmImageDitConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=GlmImageVAEConfig)

    enable_autocast: bool = False

    def get_freqs_cis(self, batch, device, rotary_emb, dtype):
        self.vae_scale_factor = 2 ** (
            len(self.vae_config.arch_config.block_out_channels) - 1
        )
        height = batch.height // self.vae_scale_factor
        width = batch.width // self.vae_scale_factor
        hidden_states = torch.empty(1, 1, height, width, device=device, dtype=dtype)
        freqs_cis = rotary_emb(hidden_states)
        return freqs_cis

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "prior_token_id": batch.prior_token_id,
            "prior_token_drop": batch.prior_token_drop_cond,
            "crop_coords": batch.crop_coords,
            "target_size": batch.target_size,
            "kv_caches": batch.kv_caches,
            "kv_caches_mode": "read",
            "freqs_cis": self.get_freqs_cis(batch, device, rotary_emb, dtype),
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "prior_token_id": batch.prior_token_id,
            "prior_token_drop": batch.prior_token_drop_uncond,
            "crop_coords": batch.crop_coords,
            "target_size": batch.target_size,
            "kv_caches": batch.kv_caches,
            "kv_caches_mode": "skip",
            "freqs_cis": self.get_freqs_cis(batch, device, rotary_emb, dtype),
        }

    def post_denoising_loop(self, latents, batch):
        # latents = latents.to(self.vae_config.dtype)
        latents_mean = (
            torch.tensor(self.vae_config.latents_mean)
            .view(1, self.vae_config.latent_channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae_config.latents_std)
            .view(1, self.vae_config.latent_channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        if getattr(batch, "kv_caches", None) is not None:
            batch.kv_caches.clear()
        return latents
