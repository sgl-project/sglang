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
"""Jina listwise rerankers (``JinaForRanking``, e.g. jina-reranker-v3.5).

One prompt holds the query and its documents. The final hidden state at each
document's embed token and at the query's rerank token goes through a small
projector, and a document's score is the cosine similarity of its projection
with the query's. The prompt is built by the /v1/rerank handler.
"""

from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_classification import Qwen3ForPooledOutput

# Ids of <|embed_token|> and <|rerank_token|> in the Jina reranker tokenizer.
DOC_EMBED_TOKEN_ID = 151670
QUERY_EMBED_TOKEN_ID = 151671
_PROJECTOR_DIM = 512


class JinaForRanking(Qwen3ForPooledOutput):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        # Qwen3Model builds full attention everywhere, but these checkpoints
        # window their sliding layers; -1 maps HF's window to window_left.
        for layer_id in range(self.model.start_layer, self.model.end_layer):
            if config.layer_types[layer_id] == "sliding_attention":
                attn = self.model.layers[layer_id].self_attn.attn
                attn.sliding_window_size = config.sliding_window - 1
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, _PROJECTOR_DIM, bias=False),
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert get_embedding, "JinaForRanking is only used for embedding"

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        seq_lens = forward_batch.extend_seq_lens_cpu
        scores = [
            self._score_documents(token_ids=token_ids, hidden_states=states)
            for token_ids, states in zip(
                torch.split(input_ids, seq_lens), torch.split(hidden_states, seq_lens)
            )
        ]
        return EmbeddingPoolerOutput(embeddings=scores)

    # Returns one cosine score per document, in prompt order.
    def _score_documents(
        self, *, token_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        doc_states = hidden_states[token_ids == DOC_EMBED_TOKEN_ID]
        query_states = hidden_states[token_ids == QUERY_EMBED_TOKEN_ID]
        if query_states.shape[0] == 0 and doc_states.shape[0] == 0:
            # Not a rerank prompt (e.g. the startup warmup request).
            return hidden_states.new_zeros(0, dtype=torch.float32)
        if query_states.shape[0] != 1 or doc_states.shape[0] == 0:
            raise ValueError(
                "A Jina rerank prompt needs one query token and at least one "
                f"document token; got {query_states.shape[0]} and {doc_states.shape[0]}"
            )
        doc_embeds = self.projector(doc_states)
        query_embed = self.projector(query_states)
        return F.cosine_similarity(doc_embeds, query_embed, dim=-1).float()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        projector_params = dict(self.projector.named_parameters())

        def backbone_weights():
            for name, weight in weights:
                if name.startswith("projector."):
                    param = projector_params[name.removeprefix("projector.")]
                    default_weight_loader(param, weight)
                else:
                    yield name, weight

        super().load_weights(backbone_weights())


EntryClass = JinaForRanking
