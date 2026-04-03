# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for ErnieImage (Text2ImgDiT)."""

from dataclasses import dataclass, field
from typing import Callable, List

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.ernie_image import ErnieImageDitConfig
from sglang.multimodal_gen.configs.models.encoders.mistral3 import Mistral3EncoderConfig
from sglang.multimodal_gen.configs.models.vaes.ernie_image import ErnieImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
    shard_rotary_emb_for_sp,
)


def ernie_image_postprocess_text(outputs, _text_inputs, hidden_layer_index=-2):
    """Extract hidden states from Mistral3 text encoder.

    Uses the second-to-last hidden state layer, matching the training setup.
    Text hiddens are returned as a list of variable-length tensors, then
    padded into a batch.
    """
    hidden_states = outputs.hidden_states[hidden_layer_index]
    # hidden_states: [B, T, H] - batch of padded text embeddings
    return hidden_states


def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """2x2 patchify: [B, 32, H, W] -> [B, 128, H/2, W/2]"""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    return latents


def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Reverse patchify: [B, 128, H/2, W/2] -> [B, 32, H, W]"""
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)
    return latents


@dataclass
class ErnieImagePipelineConfig(ImagePipelineConfig):
    """Configuration for the ErnieImage text-to-image pipeline.

    Uses a single-stream DiT with shared AdaLN, Mistral3 text encoder,
    and Flux2 VAE with BN-based latent normalization.
    """

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2I

    vae_tiling: bool = False
    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=ErnieImageDitConfig)
    vae_config: VAEConfig = field(default_factory=ErnieImageVAEConfig)

    enable_autocast: bool = False

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Mistral3EncoderConfig(),)
    )

    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16",)
    )

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (None,)  # No text preprocessing template
    )

    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (ernie_image_postprocess_text,)
    )

    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            dict(
                padding=False,
                truncation=True,
                max_length=1536,
                add_special_tokens=True,
            ),
        ]
    )

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    def get_vae_scale_factor(self):
        return 16  # 8 spatial * 2 patch

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        vae_scale_factor = self.get_vae_scale_factor()
        # Latent shape in patchified space: [B, 128, H/16, W/16]
        latent_h = batch.height // vae_scale_factor
        latent_w = batch.width // vae_scale_factor
        num_channels = self.dit_config.arch_config.in_channels  # 128
        shape = (batch_size, num_channels, latent_h, latent_w)
        return shape

    def maybe_pack_latents(self, latents, batch_size, batch):
        # ErnieImage works directly in [B, 128, H/16, W/16] space
        # No additional packing needed - latents are already patchified
        return latents

    def get_decode_scale_and_shift(self, device, dtype, vae):
        """BN denormalization via scale_and_shift (Flux-compatible pattern).

        DecodingStage applies: latents = latents / scaling_factor + shift_factor
        We return (1/bn_std, bn_mean) so that:
          latents / (1/bn_std) + bn_mean = latents * bn_std + bn_mean
        This matches the BN denormalization formula.
        """
        if hasattr(vae, 'bn') and vae.bn is not None:
            bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
            bn_var = vae.bn.running_var.view(1, -1, 1, 1).to(device, dtype)
            bn_std = torch.sqrt(bn_var + 1e-5)
            # Return reciprocal so division gives multiplication
            return 1.0 / bn_std, bn_mean
        return 1.0, None

    @staticmethod
    def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device, dtype):
        """Compute 3D RoPE frequencies for image+text sequence."""
        freqs = rotary_emb(img_shapes, txt_seq_lens, device=device)

        if isinstance(freqs, tuple) and len(freqs) == 2:
            img_freqs, txt_freqs = freqs
            img_cos = img_freqs.real.to(dtype=torch.float32).contiguous()
            img_sin = img_freqs.imag.to(dtype=torch.float32).contiguous()
            txt_cos = txt_freqs.real.to(dtype=torch.float32).contiguous()
            txt_sin = txt_freqs.imag.to(dtype=torch.float32).contiguous()
            img_cache = torch.cat([img_cos, img_sin], dim=-1)
            txt_cache = torch.cat([txt_cos, txt_sin], dim=-1)
            return img_cache, txt_cache

        # Single combined freqs
        cos = freqs.real.to(dtype=torch.float32).contiguous()
        sin = freqs.imag.to(dtype=torch.float32).contiguous()
        return torch.cat([cos, sin], dim=-1)

    def _prepare_cond_kwargs(self, batch, prompt_embeds, rotary_emb, device, dtype):
        batch_size = prompt_embeds[0].shape[0]
        height = batch.height
        width = batch.width
        vae_scale_factor = self.get_vae_scale_factor()

        img_shapes = [
            [
                (
                    1,
                    height // vae_scale_factor,
                    width // vae_scale_factor,
                )
            ]
        ] * batch_size
        txt_seq_lens = [prompt_embeds[0].shape[1]]

        if rotary_emb is None:
            return {
                "img_shapes": img_shapes,
                "txt_seq_lens": txt_seq_lens,
                "freqs_cis": None,
            }

        freqs_cis = self.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )

        if isinstance(freqs_cis, tuple):
            img_cache, txt_cache = freqs_cis
            img_cache = shard_rotary_emb_for_sp(img_cache)
            freqs_cis = (img_cache, txt_cache)

        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": freqs_cis,
            "img_shapes": img_shapes,
        }

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_cond_kwargs(
            batch, batch.prompt_embeds, rotary_emb, device, dtype
        )

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_cond_kwargs(
            batch, batch.negative_prompt_embeds, rotary_emb, device, dtype
        )

    def _check_vae_has_bn(self, vae):
        """Check if VAE has bn attribute (cached)."""
        if not hasattr(self, "_vae_has_bn_cache"):
            self._vae_has_bn_cache = hasattr(vae, "bn") and vae.bn is not None
        return self._vae_has_bn_cache

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        """Unpatchify + add frame dim before VAE decode.

        Called after scale_and_shift (BN denorm), before vae.decode.
        Unpatchify: [B, 128, H/16, W/16] -> [B, 32, H/8, W/8]
        Add frame dim: [B, 32, H/8, W/8] -> [B, 32, 1, H/8, W/8]
        """
        if vae is not None and self._check_vae_has_bn(vae):
            latents = _unpatchify_latents(latents)
        # VAE expects standard 4D input [B, C, H, W], no frame dim needed
        return latents

    def post_denoising_loop(self, latents, batch):
        """Post-denoising: no-op (unpatchify is done in preprocess_decoding)."""
        return latents
