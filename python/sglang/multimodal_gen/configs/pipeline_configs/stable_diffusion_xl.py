# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion XL pipeline configuration.

Reuses the SD3 config for CLIP tokenization (no padding mask), penultimate-layer
CLIP features, spatial latent shape, and VAE weight selection.
"""

from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.stable_diffusion_xl import (
    StableDiffusionXLUNetConfig,
)
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.vaes.stable_diffusion import (
    StableDiffusionVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.stablediffusion3 import (
    SD3CLIPTextConfig,
    StableDiffusion3PipelineConfig,
    sd3_clip_postprocess_text,
)


@dataclass
class StableDiffusionXLPipelineConfig(StableDiffusion3PipelineConfig):
    dit_config: DiTConfig = field(default_factory=StableDiffusionXLUNetConfig)
    vae_config: VAEConfig = field(default_factory=StableDiffusionVAEConfig)

    # CLIP ViT-L (text_encoder) and OpenCLIP ViT-bigG with projection (text_encoder_2).
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (SD3CLIPTextConfig(), SD3CLIPTextConfig())
    )
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp16", "fp16")
    )
    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (None, None)
    )
    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput, dict], torch.Tensor], ...
    ] = field(
        default_factory=lambda: (sd3_clip_postprocess_text, sd3_clip_postprocess_text)
    )

    guidance_scale: float = 5.0
    # The SDXL VAE overflows in fp16 (force_upcast in vae/config.json).
    vae_precision: str = "fp32"
    vae_tiling: bool = False
    vae_sp: bool = False

    def __post_init__(self) -> None:
        configs = list(self.text_encoder_configs)
        configs[0].update_model_arch({"_class_name": "CLIPTextModel"})
        configs[1].update_model_arch({"_class_name": "CLIPTextModelWithProjection"})
        self.text_encoder_configs = tuple(configs)

    def get_text_encoder_pooler_output(self, outputs, encoder_index):
        # SDXL conditions only on the projected pooled output of text_encoder_2.
        if encoder_index == 1:
            return outputs.pooler_output
        return None

    def get_pos_prompt_embeds(self, batch):
        return torch.cat(batch.prompt_embeds, dim=-1)

    def get_neg_prompt_embeds(self, batch):
        embeds = torch.cat(batch.negative_prompt_embeds, dim=-1)
        if self._zero_negative(batch):
            return torch.zeros_like(embeds)
        return embeds

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._added_cond_kwargs(batch, text_embeds=batch.pooled_embeds[0])

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        text_embeds = batch.neg_pooled_embeds[0]
        if self._zero_negative(batch):
            text_embeds = torch.zeros_like(text_embeds)
        return self._added_cond_kwargs(batch, text_embeds=text_embeds)

    @staticmethod
    def _zero_negative(batch) -> bool:
        # Diffusers' force_zeros_for_empty_prompt for SDXL base checkpoints.
        return batch.negative_prompt == ""

    @staticmethod
    def _added_cond_kwargs(batch, *, text_embeds: torch.Tensor) -> dict:
        # [original_h, original_w, crop_top, crop_left, target_h, target_w]
        time_ids = torch.tensor(
            [[batch.height, batch.width, 0, 0, batch.height, batch.width]],
            device=text_embeds.device,
            dtype=torch.float32,
        ).repeat(text_embeds.shape[0], 1)
        return {"added_cond_kwargs": {"text_embeds": text_embeds, "time_ids": time_ids}}


def register():
    from sglang.multimodal_gen.configs.sample.stable_diffusion_xl import (
        StableDiffusionXLSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=StableDiffusionXLSamplingParams,
        pipeline_config_cls=StableDiffusionXLPipelineConfig,
        hf_model_paths=["stabilityai/stable-diffusion-xl-base-1.0"],
    )
