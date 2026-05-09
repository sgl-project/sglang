from typing import Iterable, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.model_executor.model_runner import ForwardBatch
from sglang.srt.model_loader.auto_loader import AutoWeightsLoader
from sglang.srt.models.llama import LlamaModel
from sglang.srt.utils import add_prefix


class LlamaEmbeddingModel(nn.Module):
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
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

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
        ), "LlamaEmbeddingModel / MistralModel is only used for embedding"
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_substrs=["projector"],
            ignore_unexpected_prefixes=["model.vision_tower."],
        )
        return loader.load_weights(weights)


class MistralModel(LlamaEmbeddingModel):
    pass


EntryClass = [LlamaEmbeddingModel, MistralModel]
