# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Callable

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.boogu_image import BooguImageDitConfig
from sglang.multimodal_gen.configs.models.encoders.qwen3vl import (
    Qwen3VLArchConfig,
    Qwen3VLConfig,
)
from sglang.multimodal_gen.configs.models.vaes.boogu_image import BooguImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)

BOOGU_SYSTEM_PROMPT_T2I = (
    "You are a helpful assistant that generates high-quality images based on user "
    "instructions. The instructions are as follows."
)
BOOGU_SYSTEM_PROMPT_DROP = (
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate."
)


def _boogu_build_chat_messages(instruction: str) -> list[dict]:
    if instruction is None or len(instruction.strip()) == 0:
        system_prompt = BOOGU_SYSTEM_PROMPT_DROP
    else:
        system_prompt = BOOGU_SYSTEM_PROMPT_T2I
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": instruction or ""}],
        },
    ]


@dataclass
class BooguQwen3VLArchConfig(Qwen3VLArchConfig):
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_return_dict: bool = True


@dataclass
class BooguQwen3VLConfig(Qwen3VLConfig):
    arch_config: Qwen3VLArchConfig = field(default_factory=BooguQwen3VLArchConfig)


def _boogu_postprocess_text(outputs, _text_inputs):
    return outputs.last_hidden_state


@dataclass
class BooguImagePipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I

    should_use_guidance: bool = True
    enable_autocast: bool = False
    precision: str = "bf16"
    vae_precision: str = "fp32"
    vae_tiling: bool = False
    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=BooguImageDitConfig)
    vae_config: VAEConfig = field(default_factory=BooguImageVAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (BooguQwen3VLConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (_boogu_postprocess_text,)
    )

    scheduler_class_override: str = "BooguFlowMatchScheduler"

    def tokenize_prompt(self, prompts: list[str], tokenizer, tok_kwargs) -> dict:
        messages_batch = [_boogu_build_chat_messages(p) for p in prompts]
        return tokenizer.apply_chat_template(
            messages_batch,
            padding="longest",
            padding_side="right",
            truncation=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

    def get_freqs_cis(self, batch, device, rotary_emb, dtype):
        return None

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "instruction_hidden_states": batch.prompt_embeds[0].to(dtype),
            "instruction_attention_mask": batch.prompt_attention_mask[0].to(device),
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        neg_embeds = (
            batch.negative_prompt_embeds[0]
            if batch.negative_prompt_embeds is not None
            else batch.prompt_embeds[0]
        )
        neg_mask = (
            batch.negative_attention_mask[0]
            if batch.negative_attention_mask is not None
            else batch.prompt_attention_mask[0]
        )
        return {
            "instruction_hidden_states": neg_embeds.to(dtype),
            "instruction_attention_mask": neg_mask.to(device),
        }

    def post_denoising_loop(self, latents, batch):
        if latents.dim() == 5:
            latents = latents.squeeze(2)
        return latents
