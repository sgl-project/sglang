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
from transformers import Gemma2Config

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.gemma2 import Gemma2ForCausalLM, Gemma2Model


class Gemma2ForSequenceClassification(nn.Module):
    def __init__(
        self,
        config: Gemma2Config,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config, quant_config=quant_config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

        self.eos_token_id = config.eos_token_id

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert (
            get_embedding
        ), "Gemma2ForSequenceClassification is only used for embedding"

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        last_token_hidden = self.pooler(hidden_states, forward_batch).embeddings
        scores = self.score(last_token_hidden)

        return EmbeddingPoolerOutput(scores)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        Gemma2ForCausalLM.load_weights(self, weights)


EntryClass = [Gemma2ForSequenceClassification]
