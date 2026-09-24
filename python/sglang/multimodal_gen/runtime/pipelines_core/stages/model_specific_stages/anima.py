# SPDX-License-Identifier: Apache-2.0
import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.condition_encoding import (
    ConditionEncodingStage,
)


class AnimaTextConditioningStage(ConditionEncodingStage):
    def __init__(self, conditioner, tokenizer):
        super().__init__()
        self.conditioner = conditioner
        self.tokenizer = tokenizer

    def component_uses(self, server_args, stage_name=None):
        return [
            ComponentUse(self._component_stage_name(stage_name), "text_conditioner")
        ]

    @torch.no_grad()
    def forward(self, batch, server_args):
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        max_length = batch.max_sequence_length or 512
        if not 1 <= max_length <= 4096:
            raise ValueError("Anima max_sequence_length must be between 1 and 4096")
        with self.use_declared_component(
            component_name="text_conditioner", module=self.conditioner
        ) as conditioner:
            self.conditioner = conditioner
            batch.prompt_embeds = [
                self._condition(
                    conditioner,
                    prompts,
                    batch.prompt_embeds[0],
                    batch.prompt_attention_mask[0],
                    max_length,
                )
            ]
            if batch.do_classifier_free_guidance:
                negative = [batch.negative_prompt] * len(prompts)
                batch.negative_prompt_embeds = [
                    self._condition(
                        conditioner,
                        negative,
                        batch.negative_prompt_embeds[0],
                        batch.negative_attention_mask[0],
                        max_length,
                    )
                ]
        # conditioner output includes learned-token padding; Cosmos attends to it
        batch.prompt_attention_mask = None
        batch.negative_attention_mask = None
        batch.prompt_embeds_mask = [
            torch.ones_like(batch.prompt_embeds[0][..., 0], dtype=torch.bool)
        ]
        batch.prompt_seq_lens = [
            [batch.prompt_embeds[0].shape[1]] * batch.prompt_embeds[0].shape[0]
        ]
        if batch.do_classifier_free_guidance:
            batch.negative_prompt_embeds_mask = [
                torch.ones_like(
                    batch.negative_prompt_embeds[0][..., 0], dtype=torch.bool
                )
            ]
            batch.negative_prompt_seq_lens = [
                [batch.negative_prompt_embeds[0].shape[1]]
                * batch.negative_prompt_embeds[0].shape[0]
            ]
        return batch

    def _condition(self, conditioner, prompts, embeds, source_mask, max_length):
        tokens = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(embeds.device)
        embeds = embeds.to(dtype=next(conditioner.parameters()).dtype)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            return conditioner(
                embeds, tokens.input_ids, tokens.attention_mask, source_mask
            )
