# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
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

from sglang.srt.configs.qwen3_5 import Qwen3_5Config
from sglang.srt.layers.pooler import Pooler, PoolingType, score_and_pool
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration


class Qwen3_5ForSequenceClassification(Qwen3_5ForConditionalGeneration):
    """Dense Qwen3.5 with a score head over raw last-token hidden states."""

    def __init__(
        self,
        config: Qwen3_5Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            create_lm_head=False,
        )
        self.score = nn.Linear(
            config.text_config.hidden_size, config.num_labels, bias=False
        )
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

    def _pool_hidden_states(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return score_and_pool(
            score_head=self.score,
            pooler=self.pooler,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            input_ids=input_ids,
        )

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.weight if self.pp_group.is_first_rank else None
        return embed, None

    def set_embed_and_head(self, embed, head):
        # A vocabulary head must not replace the independent classification head.
        super().set_embed_and_head(embed, None)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return super().load_weights(
            (name, weight)
            for name, weight in weights
            if not name.startswith("lm_head.")
        )


EntryClass = Qwen3_5ForSequenceClassification
