# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypedDict

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import HunyuanVideoConfig
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfig,
    LlamaConfig,
)
from sglang.multimodal_gen.configs.models.vaes import HunyuanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig

PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)


class PromptTemplate(TypedDict):
    template: str
    crop_start: int


prompt_template_video: PromptTemplate = {
    "template": PROMPT_TEMPLATE_ENCODE_VIDEO,
    "crop_start": 95,
}


def llama_preprocess_text(prompt: str) -> str:
    return prompt_template_video["template"].format(prompt)


def llama_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.tensor:
    hidden_state_skip_layer = 2
    assert outputs.hidden_states is not None
    hidden_states: tuple[torch.Tensor, ...] = outputs.hidden_states
    last_hidden_state: torch.tensor = hidden_states[-(hidden_state_skip_layer + 1)]
    crop_start = prompt_template_video.get("crop_start", -1)
    last_hidden_state = last_hidden_state[:, crop_start:]
    return last_hidden_state


def clip_preprocess_text(prompt: str) -> str:
    return prompt


def clip_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.tensor:
    pooler_output: torch.tensor = outputs.pooler_output
    return pooler_output


@dataclass
class HunyuanConfig(PipelineConfig):
    """Base configuration for HunYuan pipeline architecture."""

    # HunyuanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=HunyuanVideoConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=HunyuanVAEConfig)
    # Denoising stage
    embedded_cfg_scale: int = 6
    flow_shift: int = 7

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (LlamaConfig(), CLIPTextConfig())
    )
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (llama_preprocess_text, clip_preprocess_text)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.tensor], ...] = (
        field(default_factory=lambda: (llama_postprocess_text, clip_postprocess_text))
    )

    # Precision for each component
    dit_precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp16", "fp16")
    )

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class FastHunyuanConfig(HunyuanConfig):
    """Configuration specifically optimized for FastHunyuan weights."""

    # Override HunyuanConfig defaults
    flow_shift: int = 17

    # No need to re-specify guidance_scale or embedded_cfg_scale as they
    # already have the desired values from HunyuanConfig
