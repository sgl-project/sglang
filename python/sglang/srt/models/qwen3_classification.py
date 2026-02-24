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

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config  # Qwen3 uses Qwen2Config

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Qwen3ForPooledOutput(nn.Module):
    """Base class for Qwen3 models that produce pooled output (classification, reward).

    Subclasses should set self.score and self.pooler in their __init__.
    """

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.eos_token_id = config.eos_token_id
        # Subclasses must set self.score and self.pooler

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert get_embedding, f"{self.__class__.__name__} is only used for embedding"

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        logits = self.score(hidden_states)
        pooled_logits = self.pooler(logits, forward_batch).embeddings

        return EmbeddingPoolerOutput(pooled_logits)

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
            # Skip lm_head weights (pooled output models don't have lm_head)
            if name.startswith("lm_head"):
                continue

            # Skip rotary embeddings and other non-parameter tensors
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # Handle stacked parameters (qkv_proj, gate_up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")


class Qwen3ForSequenceClassification(Qwen3ForPooledOutput):
    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.score = nn.Linear(config.hidden_size, config.num_labels)
        # Use normalize=True for qwen3 embedding based on official implementation
        # Reference: https://github.com/QwenLM/Qwen3-Embedding/blob/main/examples/qwen3_embedding_transformers.py#L55
        # Official code: output = F.normalize(output, p=2, dim=1)
        normalize = True

        # We don't want to normalize the embedding if we have a classification head
        if config.id2label is not None or config.label2id is not None:
            normalize = False

        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=normalize)


EntryClass = [
    Qwen3ForSequenceClassification,
]
