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

from sglang.srt.layers.pooler import (
    EmbeddingPoolerOutput,
    Pooler,
    PoolingType,
    score_and_pool,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration

_NO_LM_HEAD_MSG = (
    "Qwen3_5VLForSequenceClassification has no lm_head and cannot be "
    "used as a draft/target model for speculative decoding"
)


def _get_num_labels(config) -> int:
    # Same rule as llama.cpp's Qwen3_5SeqClsModel converter, so both sides agree.
    id2label = getattr(config, "id2label", None)
    if id2label:
        return len(id2label)
    return getattr(config, "num_labels", 2)


class Qwen3_5VLForSequenceClassification(Qwen3_5ForConditionalGeneration):
    """Qwen3.5-VL with lm_head replaced by a bias-free score head.

    Returns raw (unnormalized) logits of shape [num_labels] per request,
    pooled at the last token.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        # The parent rebinds self.config to text_config; the composite config
        # has no top-level hidden_size, but id2label/num_labels live on it.
        self.score = nn.Linear(
            self.config.hidden_size, _get_num_labels(config), bias=False
        )
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)
        self.lm_head = None

    def get_embed_and_head(self):
        raise NotImplementedError(_NO_LM_HEAD_MSG)

    def set_embed_and_head(self, embed, head):
        raise NotImplementedError(_NO_LM_HEAD_MSG)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = True,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> EmbeddingPoolerOutput:
        assert get_embedding, (
            "Qwen3_5VLForSequenceClassification is only used for classification"
        )

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if self.language_model_only:
            hidden_states = self.model(
                input_ids=input_ids,
                forward_batch=forward_batch,
                positions=positions,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        else:
            hidden_states = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.model,
                multimodal_model=self,
                positions=positions,
                use_deepstack=self.use_deepstack,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        if not self.pp_group.is_last_rank:
            return hidden_states
        return score_and_pool(
            self.score, self.pooler, hidden_states, forward_batch, input_ids
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # score.* goes through the parent loader as-is; only drop a stray lm_head.
        return super().load_weights(
            (name, w) for name, w in weights if not name.startswith("lm_head")
        )


EntryClass = [Qwen3_5VLForSequenceClassification]
