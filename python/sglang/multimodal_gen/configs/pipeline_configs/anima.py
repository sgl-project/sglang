# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models.dits.anima import AnimaDiTConfig
from sglang.multimodal_gen.configs.models.encoders.qwen3 import Qwen3TextConfig
from sglang.multimodal_gen.configs.models.vaes.qwenimage import QwenImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


def anima_text_output(outputs, inputs):
    return outputs.last_hidden_state * inputs.attention_mask.unsqueeze(-1)


@dataclass
class AnimaPipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I
    native_only_components: tuple[str, ...] = ("transformer", "text_conditioner")
    dit_config: AnimaDiTConfig = field(default_factory=AnimaDiTConfig)
    vae_config: QwenImageVAEConfig = field(default_factory=QwenImageVAEConfig)
    text_encoder_configs: tuple = field(default_factory=lambda: (Qwen3TextConfig(),))
    text_encoder_precisions: tuple[str, ...] = ("bf16",)
    preprocess_text_funcs: tuple[Callable | None, ...] = (None,)
    postprocess_text_funcs: tuple[Callable, ...] = (anima_text_output,)
    should_use_guidance: bool = False
    enable_autocast: bool = False
    vae_precision: str = "bf16"

    def tokenize_prompt(self, prompt, tokenizer, tok_kwargs):
        if not 1 <= tok_kwargs.get("max_length", 512) <= 4096:
            raise ValueError("Anima max_sequence_length must be between 1 and 4096")
        inputs = tokenizer(prompt, **{**tok_kwargs, "padding": "longest"})
        if inputs.input_ids.shape[1] == 0:
            inputs["input_ids"] = inputs.input_ids.new_zeros((len(prompt), 1))
            inputs["attention_mask"] = inputs.attention_mask.new_zeros((len(prompt), 1))
        return inputs

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    def get_latent_dtype(self, prompt_dtype):
        return torch.float32

    def expand_conditioning_to_sample_batch(self, batch):
        count = batch.num_outputs_per_prompt
        if count > 1:
            batch.prompt_embeds = [
                x.repeat_interleave(count, dim=0) for x in batch.prompt_embeds
            ]
            if batch.do_classifier_free_guidance:
                batch.negative_prompt_embeds = [
                    x.repeat_interleave(count, dim=0)
                    for x in batch.negative_prompt_embeds
                ]
        return batch

    # the DiT shards patch tokens, leaving scheduler latents replicated
    def shard_latents_for_sp(self, batch, latents):
        return latents, False

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def get_decode_scale_and_shift(self, device, dtype, vae):
        config = self.vae_config.arch_config
        std = torch.tensor(config.latents_std, device=device, dtype=dtype)
        mean = torch.tensor(config.latents_mean, device=device, dtype=dtype)
        return std.reciprocal().view(1, -1, 1, 1, 1), mean.view(1, -1, 1, 1, 1)

    def post_decoding(self, frames, server_args):
        return frames.squeeze(2)
