# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.zimage import ZImageDitConfig
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    TextEncoderConfig,
)
from sglang.multimodal_gen.configs.models.vaes.flux import FluxVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


def zimage_preprocess_text(prompt: str):
    messages = [
        {"role": "user", "content": prompt},
    ]
    return messages


def zimage_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    device = outputs.hidden_states[-2].device
    prompt_mask = _text_inputs.attention_mask.to(device).bool()
    return outputs.hidden_states[-2][0][prompt_mask[0]]


class TransformersModelConfig(EncoderConfig):
    tokenizer_kwargs: dict = field(default_factory=lambda: {})


@dataclass
class ZImagePipelineConfig(ImagePipelineConfig):

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2I

    dit_config: DiTConfig = field(default_factory=ZImageDitConfig)
    vae_config: VAEConfig = field(default_factory=FluxVAEConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (TextEncoderConfig(),)
    )

    preprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (zimage_preprocess_text,)
    )

    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (zimage_postprocess_text,)
    )

    def tokenize_prompt(self, prompts: list[str], tokenizer, tok_kwargs) -> dict:
        # flatten to 1-d list
        inputs = tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            padding="max_length",
            max_length=512,  # TODO (yhyang201): set max length according to config
            truncation=True,
            return_tensors="pt",
            return_dict=True,
        )
        return inputs

    def post_denoising_loop(self, latents, batch):
        bs, channels, num_frames, height, width = latents.shape
        return latents.view(bs, channels, height, width)
