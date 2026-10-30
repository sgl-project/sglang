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

from typing import Callable, Optional

import torch
from torch import nn

from sglang.srt.layers.pooler import (
    EmbeddingPoolerOutput,
    Pooler,
    PoolingType,
    score_and_pool,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration

_NO_LM_HEAD_MSG = (
    "Qwen3_5VLForSequenceClassification has no lm_head and cannot be "
    "used as a draft/target model for speculative decoding"
)


class _ScoringPooler(nn.Module):
    """Last-token pooling followed by the classification head."""

    def __init__(self, score_head: Callable[[], nn.Module]) -> None:
        super().__init__()
        # A callable, so the head stays registered once, as `score`, matching
        # the checkpoint's `score.weight`.
        self._score_head = score_head
        self._pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> EmbeddingPoolerOutput:
        # score_and_pool reads only the device from input_ids.
        return score_and_pool(
            self._score_head(),
            self._pooler,
            hidden_states,
            forward_batch,
            input_ids=hidden_states,
        )


class Qwen3_5VLForSequenceClassification(Qwen3_5ForConditionalGeneration):
    """Qwen3.5-VL with lm_head replaced by a bias-free score head.

    Returns raw (unnormalized) logits of shape [num_labels] per request,
    pooled at the last token. The parent forward runs with get_embedding=True,
    which routes the final hidden states through the scoring pooler.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        # The parent rebinds self.config to text_config; the composite config
        # has no top-level hidden_size, but id2label lives on it. len(id2label)
        # matches llama.cpp's Qwen3_5SeqClsModel converter.
        self.score = nn.Linear(
            self.config.hidden_size, len(config.id2label), bias=False
        )
        self.pooler = _ScoringPooler(lambda: self.score)
        self.lm_head = None

    def get_embed_and_head(self):
        raise NotImplementedError(_NO_LM_HEAD_MSG)

    def set_embed_and_head(self, embed, head):
        raise NotImplementedError(_NO_LM_HEAD_MSG)


class Qwen3_5ForSequenceClassification(Qwen3_5VLForSequenceClassification):
    """The Transformers name, which quantized (AWQ/GPTQ) exports declare."""

    pass


EntryClass = [Qwen3_5VLForSequenceClassification, Qwen3_5ForSequenceClassification]
