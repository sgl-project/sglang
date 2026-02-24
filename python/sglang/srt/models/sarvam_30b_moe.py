"""Inference-only SarvamMoE model compatible with HuggingFace weights for SGLang."""

from typing import Optional, Tuple

import torch

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.bailing_moe import BailingMoEForCausalLM


class SarvamMoEForCausalLM(BailingMoEForCausalLM):

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],
        input_embeds: torch.Tensor = None,
    ) -> Optional[LogitsProcessorOutput]:
        start, end = split_interval

        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.word_embeddings(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
            forward_batch.residual = None

        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            if forward_batch.residual is None:
                hidden_states = self.model.norm(forward_batch.hidden_states)
            else:
                hidden_states, _ = self.model.norm(
                    forward_batch.hidden_states, forward_batch.residual
                )
            forward_batch.hidden_states = hidden_states

            return self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )

        return None


EntryClass = [SarvamMoEForCausalLM]
