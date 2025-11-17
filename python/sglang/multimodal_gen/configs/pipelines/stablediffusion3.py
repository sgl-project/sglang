# SPDX-License-Identifier: Apache-2.0
"""StableDiffusion3 pipeline configuration."""

from dataclasses import dataclass, field
from typing import Callable, Tuple

import torch


from sglang.multimodal_gen.configs.models.dits import StableDiffusion3TransformerConfig
from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.pipelines.base import PipelineConfig
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfig,
    T5Config,
)
from sglang.multimodal_gen.configs.models.vaes.stablediffusion3 import (
    StableDiffusion3VAEConfig,
)


def t5_preprocess_text(prompt: str) -> str:
    return prompt


def clip_preprocess_text(prompt: str) -> str:
    return prompt


def clip_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.tensor:
    return outputs.hidden_states


def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    return outputs.last_hidden_state


@dataclass
class StableDiffusion3PipelineConfig(PipelineConfig):
    """Configuration for StableDiffusion3 pipeline."""

    # Model configurations - use generic ModelConfig, actual config loaded from model files
    dit_config: DiTConfig = field(default_factory=StableDiffusion3TransformerConfig)
    vae_config: VAEConfig = field(
        default_factory=StableDiffusion3VAEConfig
    )
    # Text encoders
    text_encoder_configs: Tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (
            CLIPTextConfig(), CLIPTextConfig(), T5Config()
        )
    )

    # Precision settings
    text_encoder_precisions: Tuple[str, ...] = field(
        default_factory=lambda: ("fp16", "fp16", "fp32")
    )

    # Text processing functions
    preprocess_text_funcs: Tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (
            clip_preprocess_text,  # CLIP-L
            clip_preprocess_text,  # CLIP-G
            t5_preprocess_text,  # T5-XXL
        )
    )

    postprocess_text_funcs: Tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(
        default_factory=lambda: (
            clip_postprocess_text,  # CLIP-L
            clip_postprocess_text,  # CLIP-G
            t5_postprocess_text,  # T5-XXL
        )
    )

    # SD3 specific parameters
    guidance_scale: float = 7.0

    def __post_init__(self):
        """Post initialization for SD3 specific setup."""
        self.dit_config.update_model_arch({"_class_name": "SD3Transformer2DModel"})

        configs = list(self.text_encoder_configs)
        configs[0].update_model_arch({"_class_name": "CLIPTextModelWithProjection"})
        configs[1].update_model_arch({"_class_name": "CLIPTextModelWithProjection"})
        configs[2].update_model_arch({"_class_name": "T5EncoderModel"})
        self.text_encoder_configs = tuple(configs)

    def get_pos_prompt_embeds(self, batch):
        return [batch.prompt_embeds[0], batch.pooled_embeds[0]]

    def get_neg_prompt_embeds(self, batch):
        return [batch.negative_prompt_embeds[0], batch.neg_pooled_embeds[0]]
