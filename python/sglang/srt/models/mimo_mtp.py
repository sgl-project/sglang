# Adapted from https://github.com/vllm-project/vllm/pull/17433/files  and deepseek_nextn.py

from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.mimo import MiMoForCausalLM
from sglang.srt.models.qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2Model,
)
from sglang.srt.utils import add_prefix

MiMoConfig = None


class MiMoMultiTokenPredictorLayer(nn.Module):

    def __init__(
        self,
        config: MiMoConfig,
        prefix: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()

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
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds[positions == 0] = 0
        inputs_embeds = self.token_layernorm(inputs_embeds)
        previous_hidden_states = self.hidden_layernorm(
            forward_batch.spec_info.hidden_states
        )

        hidden_states = self.input_proj(
            torch.cat([previous_hidden_states, inputs_embeds], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=None,
        )
        hidden_states = residual + hidden_states
        return self.final_layernorm(hidden_states)


class MiMoMultiTokenPredictor(nn.Module):

    def __init__(
        self,
        config: MiMoConfig,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.mtp_layers = torch.nn.ModuleDict(
            {
                str(idx): MiMoMultiTokenPredictorLayer(
                    config,
                    f"{prefix}.layers.{idx}",
                    quant_config=quant_config,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.mtp_layers[str(self.mtp_start_layer_idx + spec_step_idx)](
            input_ids,
            positions,
            forward_batch,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )


class MiMoMTP(nn.Module):
    def __init__(
        self,
        config: MiMoConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config

        self.model = MiMoMultiTokenPredictor(
            config, add_prefix("model", prefix), quant_config
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            # prefix=add_prefix("model.shared_head.head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        assert spec_step_idx == 0, "mimo_mtp only support predict one token now"
        return self.model(input_ids, positions, forward_batch)

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
        print(params_dict.keys())
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
            name = name.replace("0", "36")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mtp_layers" not in name:
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
                if "mtp_layers" not in name and (
                    "embed_tokens" not in name and "lm_head" not in name
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
        for sub_name in name_without_prefix:
            if sub_name in name:
                return name
        pattern = r"model.mtp_layers.(\d+)."
        group = re.match(pattern, name)
        if group is not None:
            name = name.replace(group.group(), group.group() + "mtp_block.")
        return name


EntryClass = MiMoMTP
