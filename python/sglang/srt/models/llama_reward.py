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

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM, LlamaModel


class LlamaForSequenceClassification(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.torchao_config = None
        self.quant_config = quant_config
        self.num_labels = config.num_labels
        self.model = LlamaModel(config, quant_config=quant_config)
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
        ), "LlamaForSequenceClassification is only used for embedding"

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        last_token_hidden = self.pooler(hidden_states, forward_batch).embeddings
        scores = self.score(last_token_hidden)

        return EmbeddingPoolerOutput(scores)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return LlamaForCausalLM.load_weights(self, weights)


class LlamaForSequenceClassificationWithNormal_Weights(LlamaForSequenceClassification):
    class Weights(torch.nn.Module):
        def __init__(self, hidden_size, num_label):
            super().__init__()
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float16),
                torch.nn.SELU(),
                torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float16),
                torch.nn.SELU(),
                torch.nn.Linear(hidden_size, num_label // 2, dtype=torch.float16),
            )

        def forward(self, x):
            return self.fc(x.to(torch.float16))

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config, quant_config)
        self.weights = self.Weights(config.hidden_size, self.num_labels)

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
        ), "LlamaForSequenceClassification is only used for embedding"
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        logits = self.score(hidden_states)
        weights = self.weights(hidden_states)

        pooled_logits = self.pooler(logits, forward_batch).embeddings
        pooled_weights = self.pooler(weights, forward_batch).embeddings

        rews = pooled_logits.view(-1, self.num_labels // 2, 2)[:, :, 0].view(
            -1, self.num_labels // 2
        )
        scores = (rews * pooled_weights).sum(dim=-1).view(-1, 1)
        return EmbeddingPoolerOutput(scores)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return super().load_weights(weights)


EntryClass = [
    LlamaForSequenceClassification,
    LlamaForSequenceClassificationWithNormal_Weights,
]
