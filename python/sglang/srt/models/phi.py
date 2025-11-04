# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/phi.py
from typing import Iterable, Optional

import torch
from torch import nn
from transformers import PhiConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers


class PhiAttention(nn.Module):

    def __init__(
        self,
        config: PhiConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id: int = 0,
    ):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads

        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_size,
            self.total_num_heads,
            bias=True,
            quant_config=quant_config,
        )
        self.dense = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            quant_config=quant_config,
        )

        scaling = self.head_size**-0.5
        rotary_dim = int(
            config.partial_rotary_factor
            * (config.hidden_size // config.num_attention_heads)
        )
        assert rotary_dim % 2 == 0

        rope_theta = getattr(config, "rope_theta", 10000.0)
        max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
        self.rotary_emb = get_rope(
            self.head_size,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_size,
            scaling,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v, forward_batch=forward_batch)
        output, _ = self.dense(attn_output)
        return output


class PhiMLP(nn.Module):

    def __init__(
        self, config: PhiConfig, quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()

        n_inner = getattr(config, "n_inner", None)
        n_inner = n_inner if n_inner is not None else 4 * config.hidden_size

        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            n_inner,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            n_inner,
            config.hidden_size,
            quant_config=quant_config,
        )
        self.act = get_act_fn(config.hidden_act)

    def forward(self, hidden_states):
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class PhiLayer(nn.Module):

    def __init__(
        self,
        config: PhiConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        idx: int = 0,
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.self_attn = PhiAttention(
            config,
            quant_config,
            prefix=add_prefix("self_attn", prefix),
            layer_id=idx,
        )
        self.mlp = PhiMLP(config, quant_config)

    def forward(
        self,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        return hidden_states


class PhiModel(nn.Module):

    def __init__(
        self,
        config: PhiConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        pp_group = get_pp_group()
        pp_size = pp_group.world_size
        pp_rank = pp_group.rank

        self.start_layer = pp_rank * config.num_hidden_layers // pp_size
        self.end_layer = (pp_rank + 1) * config.num_hidden_layers // pp_size

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: PhiLayer(
                config, quant_config=quant_config, prefix=prefix, idx=idx
            ),
            prefix=add_prefix("layers", prefix),
        )

        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]

            hidden_states = layer(
                position_ids=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
            )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class PhiForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    def __init__(
        self,
        config: PhiConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = PhiModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:

        hidden_states = self.model(
            input_ids=input_ids,
            forward_batch=forward_batch,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        weights = dict(weights)
        loaded_keys = set()

        for name, param in params_dict.items():
            if name in loaded_keys:
                continue

            # Handle packed weights
            is_packed = False
            for packed_name, src_names in self.packed_modules_mapping.items():
                if packed_name not in name:
                    continue

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                for src_name in src_names:
                    full_src_name = name.replace(packed_name, src_name)
                    if full_src_name in weights:
                        loaded_weight = weights[full_src_name]
                        # The shard_id for QKVParallelLinear is 'q', 'k', 'v'.
                        shard_id = src_name.split("_")[0]
                        weight_loader(param, loaded_weight, shard_id)
                        loaded_keys.add(full_src_name)

                loaded_keys.add(name)
                is_packed = True
                break
            if is_packed:
                continue

            # Handle non-packed weights
            if name not in weights:
                # Redundant with the check in the loop, but good for safety
                continue

            loaded_weight = weights[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_keys.add(name)


EntryClass = PhiForCausalLM
