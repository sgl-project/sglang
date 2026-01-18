# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper around `atom` models"""
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput

logger = logging.getLogger(__name__)


class ATOMForCausalLM(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        logger.info("Using Atom backend.")

        self.quant_config = quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size

        import atom
        self.model = atom.prepare_model(config=config, framework="sglang")
        if self.model is None:
            model_arch = config.model_config.architectures[0]
            raise ValueError(f'This model{model_arch} is not supported by atom')

        self.logits_processor = LogitsProcessor(config)


    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        aux_hidden_states = None
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=input_embeds,
            forward_batch=forward_batch,
            get_embedding=get_embedding,
            pp_proxy_tensors=None,
        )

        return self.logits_processor(
            input_ids, hidden_states, self.model.lm_head, forward_batch, aux_hidden_states
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self.model.load_weights(weights)

EntryClass = [ATOMForCausalLM]
