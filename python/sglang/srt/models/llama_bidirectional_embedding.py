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
"""
Support for LlamaBidirectionalModel embedding architecture.

This model is used by nvidia/llama-embed-nemotron-8b and similar models that
use bidirectional (non-causal) attention with mean pooling for text embeddings.

Key differences from LlamaEmbeddingModel:
  - Uses ENCODER_ONLY attention (bidirectional, non-causal)
  - Uses mean pooling instead of last-token pooling
"""

from typing import Iterable, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.model_runner import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaModel
from sglang.srt.utils import add_prefix


class LlamaBidirectionalModel(nn.Module):
    """
    Llama-based bidirectional embedding model.

    Uses non-causal (encoder-only) attention and mean pooling, matching the
    architecture of nvidia/llama-embed-nemotron-8b.
    """

    def __init__(
        self,
        config: LlamaConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.model = LlamaModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # Override attention layers to use bidirectional (non-causal) attention
        for layer in self.model.layers:
            layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY

        # Read pooling config from model config (defaults to mean)
        pooling = getattr(config, "pooling", "avg")
        if pooling in ("avg", "mean"):
            pooling_type = PoolingType.MEAN
        elif pooling == "last":
            pooling_type = PoolingType.LAST
        elif pooling == "cls":
            pooling_type = PoolingType.CLS
        else:
            pooling_type = PoolingType.MEAN

        self.pooler = Pooler(pooling_type=pooling_type, normalize=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert get_embedding, "LlamaBidirectionalModel is only used for embedding"
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.model.named_parameters())

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "projector" in name:
                return
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                return
            if name.startswith("model.vision_tower") and name not in params_dict:
                return

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    return
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = LlamaBidirectionalModel
