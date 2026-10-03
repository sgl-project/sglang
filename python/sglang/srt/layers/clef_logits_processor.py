"""Run Clef's joint head while preserving ordinary generation logits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.aux_hidden_states import AuxHiddenStates
from sglang.srt.layers.clef import forward_clef
from sglang.srt.layers.clef_reference import JointSchemaHead
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.auxiliary_output import append_auxiliary_output

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class ClefLogitsProcessor(LogitsProcessor):
    def __init__(self, config: PretrainedConfig, head: JointSchemaHead) -> None:
        super().__init__(config)
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: ForwardBatch,
        aux_hidden_states: AuxHiddenStates | None = None,
        hidden_states_before_norm: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        decision_output = forward_clef(
            self.head, lm_head, input_ids, hidden_states, logits_metadata
        )
        output = super().forward(
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            aux_hidden_states,
            hidden_states_before_norm,
        )
        if decision_output is not None:
            output.auxiliary_device_output = append_auxiliary_output(
                output.auxiliary_device_output, decision_output
            )
        return output
