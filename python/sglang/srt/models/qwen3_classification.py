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

from sglang.srt.layers.pooler import (
    EmbeddingPoolerOutput,
    Pooler,
    PoolingType,
    score_and_pool,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.auto_loader import AutoWeightsLoader
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

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

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
        return score_and_pool(
            self.score, self.pooler, hidden_states, forward_batch, input_ids
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Pooled-output checkpoints sometimes ship an unused ``lm_head`` tensor;
        # drop it. ``projector`` stays as a skip substring for parity with the
        # other Qwen3 loaders. Stacked QKV / gate_up are handled by the inner
        # ``Qwen3Model.load_weights`` (inherited from ``Qwen2Model``).
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["lm_head."],
            skip_substrs=["projector"],
        )
        return loader.load_weights(weights)


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
