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
"""Bidirectional Mistral3 multimodal sequence classification."""

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, Mistral3Config

from sglang.srt.configs.nemotron35_reranker import is_nemotron35_reranker_config
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.models.ministral3 import Ministral3Model
from sglang.srt.models.mistral import Mistral3ForConditionalGeneration
from sglang.srt.utils import add_prefix


class Ministral3ForSequenceClassification(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Ministral3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        for layer in self.model.layers[self.model.start_layer : self.model.end_layer]:
            layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY
        self.is_last_rank = self.model.pp_group.is_last_rank
        self.score = nn.Linear(config.hidden_size, 1, bias=False, dtype=torch.float32)
        self.pooler = Pooler(pooling_type=PoolingType.MEAN, normalize=False)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = True,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> EmbeddingPoolerOutput | PPProxyTensors:
        if not get_embedding:
            raise ValueError("Mistral3 sequence classification requires embedding mode")
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self.is_last_rank:
            return hidden_states
        # Accumulate packed sequences in FP32 so earlier requests do not affect pooling.
        pooled = self.pooler(hidden_states.float(), forward_batch).embeddings
        pooled = pooled.to(hidden_states.dtype)
        scores = self.score(pooled.float())
        return EmbeddingPoolerOutput(
            embeddings=scores,
            pooled_hidden_states=(
                pooled if forward_batch.return_pooled_hidden_states else None
            ),
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return LlamaForCausalLM._legacy_load_weights(self, weights)


class Mistral3ForSequenceClassification(Mistral3ForConditionalGeneration):
    def __init__(
        self,
        config: Mistral3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        if not is_nemotron35_reranker_config(config.to_dict()):
            raise ValueError(
                "Mistral3 reranking requires a Ministral3/Pixtral model with "
                "bidirectional attention, mean pooling and one score"
            )
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

    def _get_sgl_model_cls(self, config, auto_model_type=AutoModel):
        if auto_model_type is AutoModelForCausalLM:
            return Ministral3ForSequenceClassification
        return super()._get_sgl_model_cls(config, auto_model_type)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        def normalize():
            for name, weight in weights:
                if name == "score.weight":
                    name = "language_model.score.weight"
                yield name, weight

        return super().load_weights(normalize())


EntryClass = Mistral3ForSequenceClassification
