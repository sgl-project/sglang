# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Callable

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
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def ernie_image_postprocess_text(outputs, _text_inputs, hidden_layer_index=-2):
    hidden_states = outputs.hidden_states[hidden_layer_index]
    return hidden_states


def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    return latents


def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)
    return latents


@dataclass
class ErnieImagePipelineConfig(ImagePipelineConfig):
    """Configuration for the ErnieImage text-to-image pipeline."""

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2I

    pe_model_max_length: int = None

    vae_tiling: bool = False
    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=ErnieImageDitConfig)
    vae_config: VAEConfig = field(default_factory=ErnieImageVAEConfig)

    enable_autocast: bool = False

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Mistral3EncoderConfig(),)
    )

    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (None,)
    )

    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (ernie_image_postprocess_text,)
    )

    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            dict(
                padding=False,
                truncation=True,
                max_length=None,
                add_special_tokens=True,
            ),
        ]
    )

    def tokenize_prompt(self, prompt: list[str], tokenizer, tok_kwargs) -> dict:
        max_length = tok_kwargs.get("max_length")
        if max_length is not None:
            check = tokenizer(
                prompt,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=tok_kwargs.get("add_special_tokens", True),
            )
            for i, ids in enumerate(check["input_ids"]):
                if ids.shape[-1] > max_length:
                    logger.warning(
                        "Prompt #%d has %d tokens, exceeds max_length=%d. "
                        "The tail will be silently truncated.",
                        i,
                        ids.shape[-1],
                        max_length,
                    )
        return tokenizer(prompt, **tok_kwargs)

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    def get_vae_scale_factor(self):
        return 16

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        vae_scale_factor = self.get_vae_scale_factor()
        latent_h = batch.height // vae_scale_factor
        latent_w = batch.width // vae_scale_factor
        num_channels = self.dit_config.arch_config.in_channels  # 128
        shape = (batch_size, num_channels, latent_h, latent_w)
        return shape

    def maybe_pack_latents(self, latents, batch_size, batch):
        return latents

    def get_decode_scale_and_shift(self, device, dtype, vae):
        if hasattr(vae, "bn") and vae.bn is not None:
            bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
            bn_var = vae.bn.running_var.view(1, -1, 1, 1).to(device, dtype)
            bn_std = torch.sqrt(bn_var + 1e-5)
            return 1.0 / bn_std, bn_mean
        return 1.0, None

    @staticmethod
    def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device, dtype):
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
        if not hasattr(self, "_vae_has_bn_cache"):
            self._vae_has_bn_cache = hasattr(vae, "bn") and vae.bn is not None
        return self._vae_has_bn_cache

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        if vae is not None and self._check_vae_has_bn(vae):
            latents = _unpatchify_latents(latents)
        return latents

    def post_denoising_loop(self, latents, batch):
        return latents
