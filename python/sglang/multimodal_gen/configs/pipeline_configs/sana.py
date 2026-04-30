# SPDX-License-Identifier: Apache-2.0
#
# Pipeline configuration for SANA text-to-image generation.
#
# SANA produces 4D spatial latents (B, C, H', W') directly — unlike Flux/QwenImage
# which use packed token-style latents (B, S, D). This means:
#   - We inherit SpatialImagePipelineConfig (not ImagePipelineConfig)
#   - prepare_latent_shape returns 4D, not 5D
#   - post_denoising_loop is a no-op (no un-packing needed)
#   - shard_latents_for_sp shards along the H' dimension
#
# SANA does NOT use rotary position embeddings, so prepare_pos/neg_cond_kwargs
# return empty dicts (the DiT only needs hidden_states + encoder_hidden_states + timestep).
#
# CFG is handled by the denoising stage via guidance_scale in sampling params.
# should_use_guidance=False means no embedded guidance (no extra guidance token in forward),
# but negative_prompt + guidance_scale > 1.0 still enables standard classifier-free guidance.

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.sana import SanaConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.base import EncoderConfig
from sglang.multimodal_gen.configs.models.encoders.gemma2 import Gemma2Config
from sglang.multimodal_gen.configs.models.vaes.sana import SanaVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
)


def sana_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    # SANA uses the final hidden state from Gemma2 directly as text conditioning.
    # No intermediate-layer extraction or masking needed (unlike QwenImage/ZImage).
    return outputs.last_hidden_state


@dataclass
class SanaPipelineConfig(SpatialImagePipelineConfig):

    task_type: ModelTaskType = ModelTaskType.T2I

    # should_use_guidance=False disables *embedded* guidance (timestep-conditioned
    # guidance token). Standard CFG via guidance_scale is still active.
    should_use_guidance: bool = False
    enable_autocast: bool = False

    # DC-AE does not support tiling or SP VAE decode yet.
    vae_tiling: bool = False
    vae_sp: bool = False
    vae_precision: str = "bf16"

    dit_config: DiTConfig = field(default_factory=SanaConfig)
    vae_config: VAEConfig = field(default_factory=SanaVAEConfig)

    # Single text encoder: Gemma2 (unlike Flux which uses CLIP + T5)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Gemma2Config(),)
    )

    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (None,),
    )

    postprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (sana_postprocess_text,)
    )

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        # 4D latent shape: (B, C, H', W') — no temporal dim for T2I.
        # DC-AE compresses 1024x1024 -> 32x32 with 32 channels.
        compression = self.vae_config.arch_config.spatial_compression_ratio
        height = batch.height // compression
        width = batch.width // compression
        num_channels = self.dit_config.arch_config.num_channels_latents
        shape = (batch_size, num_channels, height, width)
        return shape

    def get_pos_prompt_embeds(self, batch):
        # Single encoder -> index [0] (Flux uses [1] because T5 is encoder #2)
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        # encoder_attention_mask: batch stores list-of-tensors; diffusers' SanaTransformer
        # expects a single tensor (sglang's has list handling). Override with [0].
        out = {}
        m = batch.prompt_attention_mask
        if isinstance(m, (list, tuple)):
            out["encoder_attention_mask"] = m[0] if m else None
        elif m is not None:
            out["encoder_attention_mask"] = m
        return out

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        out = {}
        m = batch.negative_attention_mask
        if isinstance(m, (list, tuple)):
            out["encoder_attention_mask"] = m[0] if m else None
        elif m is not None:
            out["encoder_attention_mask"] = m
        return out

    def post_denoising_loop(self, latents, batch):
        return latents

    def shard_latents_for_sp(self, batch, latents):
        # Sana's DiT uses local attention kernels and does not preserve semantics
        # when spatial latents are sequence-sharded.
        return latents, False

    def gather_latents_for_sp(self, latents):
        return latents
