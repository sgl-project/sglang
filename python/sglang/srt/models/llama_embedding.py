from typing import Iterable, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.model_executor.model_runner import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
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
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                return
            if name.startswith("model.vision_tower") and name not in params_dict:
                return

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    return
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


class MistralModel(LlamaEmbeddingModel):
    pass


EntryClass = [LlamaEmbeddingModel, MistralModel]
