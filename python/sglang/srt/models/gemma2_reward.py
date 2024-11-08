"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import Gemma2Config
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.gemma2 import Gemma2ForCausalLM, Gemma2Model


class Gemma2ForSequenceClassification(nn.Module):
    def __init__(
        self,
        config: Gemma2Config,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.torchao_config = None
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
        scores = self.score(hidden_states)

        return self.pooler(scores, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # lm_head is not used in vllm as it is tied with embed_token.
                # To prevent errors, skip loading lm_head.weight.
                if "lm_head.weight" in name:
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        Gemma2ForCausalLM.load_weights(self, weights)


EntryClass = [Gemma2ForSequenceClassification]
