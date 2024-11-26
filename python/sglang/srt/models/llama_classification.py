# Copyright 2023-2024 SGLang Team
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

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaForCausalLM, LlamaModel


class LlamaForClassification(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.torchao_config = None
        self.quant_config = quant_config
        self.model = LlamaModel(config, quant_config=quant_config)

        self.classification_head = nn.Linear(
            config.hidden_size, config.classification_out_size, bias=False
        )
        self.eos_token_id = config.eos_token_id

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        is_eos_token = input_ids == self.eos_token_id
        hidden_states = hidden_states[is_eos_token]
        scores = self.classification_head(hidden_states)

        if scores.shape[0] != forward_batch.batch_size:
            print("Warning: the EOS tokens are missing in some sentences.")
            scores = torch.ones(
                (forward_batch.batch_size, self.config.classification_out_size)
            ).to(input_ids.device)

        logits_output = LogitsProcessorOutput(
            next_token_logits=scores,
            next_token_logprobs=scores,
            normalized_prompt_logprobs=scores,
            input_token_logprobs=torch.ones_like(input_ids),
            input_top_logprobs=None,
            output_top_logprobs=None,
        )

        return logits_output

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if "classification_head" in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            elif "lm_head" in name:
                continue
            else:
                LlamaForCausalLM.load_weights(self, [(name, loaded_weight)])


EntryClass = LlamaForClassification
