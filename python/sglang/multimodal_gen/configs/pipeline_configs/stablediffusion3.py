# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion 3 pipeline configuration."""

import os
from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import StableDiffusion3TransformerConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.base import TextEncoderArchConfig
from sglang.multimodal_gen.configs.models.encoders.clip import (
    CLIPTextArchConfig,
    CLIPTextConfig,
)
from sglang.multimodal_gen.configs.models.encoders.t5 import (
    T5ArchConfig,
    T5Config,
)
from sglang.multimodal_gen.configs.models.vaes.stablediffusion3 import (
    StableDiffusion3VAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
)


def sd3_clip_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    """Extract pre-final hidden state for SD3 CLIP encoders."""
    if outputs.hidden_states is None:
        raise ValueError(
            "SD3 CLIP postprocessing requires hidden_states from encoder output."
        )
    return outputs.hidden_states[-2]


def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    return outputs.last_hidden_state


def select_sd3_vae_weight_files(
    safetensors_list: list[str],
    component_model_path: str,
    component_name: str,
    vae_precision: str,
) -> list[str]:
    """Select SD3 VAE checkpoint file candidates with minimal policy."""
    if component_name not in ("vae", "video_vae"):
        return safetensors_list

    base_name = "diffusion_pytorch_model"
    if vae_precision == "fp16":
        fp16_path = os.path.join(component_model_path, f"{base_name}.fp16.safetensors")
        if os.path.exists(fp16_path):
            return [fp16_path]

    full_path = os.path.join(component_model_path, f"{base_name}.safetensors")
    if os.path.exists(full_path):
        return [full_path]
    return safetensors_list


@dataclass
class SD3CLIPTextArchConfig(CLIPTextArchConfig):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.tokenizer_kwargs.update(
            {
                "max_length": self.text_len,
                "padding": "max_length",
            }
        )


@dataclass
class SD3CLIPTextConfig(CLIPTextConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=SD3CLIPTextArchConfig)


@dataclass
class SD3T5ArchConfig(T5ArchConfig):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.tokenizer_kwargs.update({"max_length": 256})


@dataclass
class SD3T5Config(T5Config):
    arch_config: TextEncoderArchConfig = field(default_factory=SD3T5ArchConfig)


@dataclass
class StableDiffusion3PipelineConfig(SpatialImagePipelineConfig):
    """Configuration for SD3 image generation pipeline.

    This config intentionally relies on SD3-specific encoder configs to provide
    tokenizer kwargs, instead of stage-level tokenizer overrides.
    """

    task_type: ModelTaskType = ModelTaskType.T2I

    dit_config: DiTConfig = field(default_factory=StableDiffusion3TransformerConfig)
    vae_config: VAEConfig = field(default_factory=StableDiffusion3VAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (
            SD3CLIPTextConfig(),
            SD3CLIPTextConfig(),
            SD3T5Config(),
        )
    )

    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp16", "fp16", "fp32")
    )

    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (
            None,
            None,
            None,
        )
    )

    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput, dict], torch.Tensor], ...
    ] = field(
        default_factory=lambda: (
            sd3_clip_postprocess_text,
            sd3_clip_postprocess_text,
            t5_postprocess_text,
        )
    )

    should_use_guidance: bool = False
    guidance_scale: float = 7.0

    def __post_init__(self) -> None:
        configs = list(self.text_encoder_configs)
        configs[0].update_model_arch({"_class_name": "CLIPTextModelWithProjection"})
        configs[1].update_model_arch({"_class_name": "CLIPTextModelWithProjection"})
        configs[2].update_model_arch({"_class_name": "T5EncoderModel"})
        self.text_encoder_configs = tuple(configs)

    def get_text_encoder_pooler_output(self, outputs, encoder_index):
        # SD3 uses pooled embeddings only from the two CLIP encoders (indices 0 and 1).
        if encoder_index <= 1:
            return outputs.pooler_output
        return None

    def select_vae_weight_files(
        self,
        safetensors_list: list[str],
        component_model_path: str,
        component_name: str,
        vae_precision: str,
    ) -> list[str]:
        return select_sd3_vae_weight_files(
            safetensors_list=safetensors_list,
            component_model_path=component_model_path,
            component_name=component_name,
            vae_precision=vae_precision,
        )

    def tokenize_prompt(self, prompt: list[str], tokenizer, tok_kwargs) -> dict:
        text_inputs = tokenizer(prompt, **tok_kwargs)
        text_inputs["attention_mask"] = None
        return text_inputs

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "pooled_projections": (
                batch.pooled_embeds[0] if batch.pooled_embeds else None
            )
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "pooled_projections": (
                batch.neg_pooled_embeds[0] if batch.neg_pooled_embeds else None
            )
        }

    # SD3 image latents are spatial (B, C, H, W), not video-like (B, C, T, H, W).
    def prepare_latent_shape(self, batch, batch_size, num_frames):  # noqa: ARG002
        spatial_ratio = self.vae_config.arch_config.spatial_compression_ratio
        in_channels = self.dit_config.arch_config.in_channels
        return (
            batch_size,
            in_channels,
            batch.height // spatial_ratio,
            batch.width // spatial_ratio,
        )
