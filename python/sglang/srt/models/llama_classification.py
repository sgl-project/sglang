from typing import Iterable, Optional, Tuple

import torch
import tqdm
from torch import nn
from transformers import LlamaConfig
from vllm.config import CacheConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.managers.controller.model_runner import InputMetadata
from sglang.srt.layers.logits_processor import LogitProcessorOutput
from sglang.srt.models.llama2 import LlamaModel


class LlamaForClassification(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = LlamaModel(config, quant_config=quant_config)

        self.classification_head = nn.Linear(config.hidden_size, config.classification_out_size)
        self.eos_token_id = config.eos_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, input_metadata, input_embeds)
        is_eos_token = input_ids == self.eos_token_id
        hidden_states = hidden_states[is_eos_token]
        scores = self.classification_head(hidden_states)

        if scores.shape[0] != input_metadata.batch_size:
            print("Warning: the EOS tokens are missing in some sentences.")
            scores = torch.ones((input_metadata.batch_size, self.config.classification_out_size)).to(input_ids.device)

        return LogitProcessorOutput(
            next_token_logits=scores,
            next_token_logprobs=scores,
            normalized_prompt_logprobs=scores,
            prefill_token_logprobs=torch.ones_like(input_ids),
            prefill_top_logprobs=None,
            decode_top_logprobs=None,
        )

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
        if get_tensor_model_parallel_rank() == 0:
            weights = tqdm.tqdm(weights, total=int(len(params_dict) * 1.5))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if "lm_head" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.startswith("model.vision_tower") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.startswith("model.vision_tower") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

EntryClass = LlamaForClassification