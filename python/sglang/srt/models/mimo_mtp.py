# Adapted from https://github.com/vllm-project/vllm/pull/17433/files  and deepseek_nextn.py

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2DecoderLayer


class MiMoMultiTokenPredictorLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.token_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.mtp_block = Qwen2DecoderLayer(
            config=config, quant_config=quant_config, prefix=prefix
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        # masking inputs at position 0, as not needed by MTP
        hidden_states[positions == 0] = 0

        hidden_states = self.input_proj(
            torch.cat(
                (
                    self.hidden_layernorm(forward_batch.spec_info.hidden_states),
                    self.token_layernorm(hidden_states),
                ),
                dim=-1,
            )
        )

        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=None,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class MiMoMTP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config

        self.model = MiMoMultiTokenPredictorLayer(
            config,
            prefix,
            quant_config,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
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
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            name = self.map_model_name_to_mtp_param_name(name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mtp_block" not in name:
                    break
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
                    continue
                if "mtp_block" not in name and (
                    "embed_tokens" not in name
                    and "lm_head" not in name
                    and "token_layernorm" not in name
                    and "hidden_layernorm" not in name
                    and "input_proj" not in name
                    and "final_layernorm" not in name
                ):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def map_model_name_to_mtp_param_name(self, name: str) -> str:
        import re

        name_without_prefix = [
            "token_layernorm",
            "hidden_layernorm",
            "input_proj",
            "final_layernorm",
        ]
        pattern = r"model.mtp_layers.(\d+)."
        group = re.match(pattern, name)
        if group is not None:
            for sub_name in name_without_prefix:
                if sub_name in name:
                    name = name.replace(group.group(), "model.")
                    return name
            name = name.replace(group.group(), "model.mtp_block.")
        return name

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = MiMoMTP
