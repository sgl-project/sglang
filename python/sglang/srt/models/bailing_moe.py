# Copyright 2023-2024 SGLang Team
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/bailing_moe.py

from collections.abc import Iterable
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
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


class BailingAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads

        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim

        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=quant_config,
            prefix=add_prefix("query_key_value", prefix),
        )

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=add_prefix("dense", prefix),
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
            rope_scaling=config.rope_scaling,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self.rotary_emb(position_ids, q, k)
        context_layer = self.attn(q, k, v, forward_batch)
        attn_output, _ = self.dense(context_layer)
        return attn_output


class BailingMLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: Optional[bool] = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class BailingMoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.num_shared_experts
        self.norm_expert_prob = config.norm_topk_prob
        self.moe_intermediate_size = config.moe_intermediate_size

        self.gate = ReplicatedLinear(
            self.hidden_size, self.num_experts, bias=False, quant_config=None
        )

        self.topk = TopK(top_k=self.top_k, renormalize=self.norm_expert_prob)

        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            layer_id=layer_id,
            hidden_size=self.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )

        if self.num_shared_experts > 0:
            shared_intermediate_size = (
                self.moe_intermediate_size * self.num_shared_experts
            )
            self.shared_experts = BailingMLP(
                intermediate_size=shared_intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)

        shared_output = None
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states_flat)

        router_logits, _ = self.gate(hidden_states_flat)
        topk_output = self.topk(hidden_states_flat, router_logits)
        final_hidden_states = self.experts(hidden_states_flat, topk_output)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(orig_shape)


class BailingMoeBlock(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = BailingAttention(
            config, layer_id, quant_config, prefix=add_prefix("attention", prefix)
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = BailingMoE(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-normalization and residual connection for the attention block
        if residual is None:
            residual = hidden_states
            normed_hidden_states = self.input_layernorm(hidden_states)
        else:
            normed_hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )

        attn_output = self.attention(
            hidden_states=normed_hidden_states,
            position_ids=position_ids,
            forward_batch=forward_batch,
        )

        # Pre-normalization and residual connection for the MLP block
        normed_hidden_states, residual = self.post_attention_layernorm(
            attn_output, residual
        )
        mlp_output = self.mlp(normed_hidden_states)

        return mlp_output, residual


class BailingMoeModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.embedding_dropout = torch.nn.Dropout(config.embedding_dropout)

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: BailingMoeBlock(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )

        self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                position_ids,
                residual,
                forward_batch,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class BailingMoeForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = BailingMoeModel(config=config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            quant_config=quant_config,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:

            if (
                hasattr(self.config, "norm_head")
                and self.config.norm_head
                and "lm_head.weight" in name
            ):
                loaded_weight = F.normalize(loaded_weight, dim=0, p=2, eps=1e-7)

            if "model.word_embeddings.weight" == name:
                name = "model.embed_tokens.weight"

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name and "mlp.experts" not in name:
                    full_param_name = name.replace(weight_name, param_name)
                    param = params_dict[full_param_name]
                    param.weight_loader(param, loaded_weight, shard_id)
                    break
            else:
                for p_name, w_name, e_id, s_id in expert_params_mapping:
                    if w_name in name and "mlp.experts" in name:
                        full_param_name = name.replace(w_name, p_name)
                        param = params_dict[full_param_name]
                        param.weight_loader(
                            param,
                            loaded_weight,
                            full_param_name,
                            shard_id=s_id,
                            expert_id=e_id,
                        )
                        break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


EntryClass = BailingMoeForCausalLM
