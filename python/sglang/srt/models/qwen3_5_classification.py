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

logger = logging.getLogger(__name__)


class Qwen3_5VLForSequenceClassification(Qwen3_5ForConditionalGeneration):
    """Qwen3.5-VL fine-tuned as a sequence classifier.

    score.weight [num_labels, hidden_size] replaces lm_head; lm_head is unused.
    Served with --is-embedding; /encode returns the raw classification logits.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        num_labels = getattr(config, "num_labels", 2)
        self.score = nn.Linear(config.hidden_size, num_labels, bias=False)
        # LAST-token pooling without L2 normalization — logits, not embeddings.
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)
        # No lm_head in this checkpoint; clear the one built by the parent.
        self.lm_head = None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = True,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> EmbeddingPoolerOutput:
        assert get_embedding, "Qwen3_5VLForSequenceClassification is only used for classification"

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

        return score_and_pool(self.score, self.pooler, hidden_states, forward_batch, input_ids)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Materialize once so we can handle score.* before delegating to parent.
        weights = list(weights)
        params_dict = dict(self.named_parameters())

        remaining = []
        for name, loaded_weight in weights:
            if name.startswith("lm_head"):
                # No lm_head in this checkpoint; skip silently.
                continue
            if name.startswith("score."):
                if name in params_dict:
                    from sglang.srt.model_loader.weight_utils import default_weight_loader
                    weight_loader = getattr(params_dict[name], "weight_loader", default_weight_loader)
                    weight_loader(params_dict[name], loaded_weight)
                else:
                    logger.warning(f"score weight {name} not found in params_dict")
                continue
            remaining.append((name, loaded_weight))

        return super().load_weights(remaining)


EntryClass = [Qwen3_5VLForSequenceClassification]
