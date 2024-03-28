# Adapted from
# https://github.com/vllm-project/vllm/blob/c81dddb45c71e630b907f9d84686ecd73b4105c7/vllm/model_executor/models/gpt_neox.py#L1
"""Inference-only GPT-NeoX model compatible with HuggingFace weights."""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import GPTNeoXConfig

from vllm.model_executor.layers.activation import get_act_fn
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.router.model_runner import InputMetadata
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)

class GPTNeoXAttention(nn.Module):

    def __init__(
        self,
        config: GPTNeoXConfig,
        layer_id: int = 0,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads
        self.bias = getattr(config, "attention_bias", True)

        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)

        self.query_key_value = QKVParallelLinear(
            config.hidden_size,
            self.head_size,
            self.total_num_heads,
            bias=self.bias,
            linear_method=linear_method,
        )
        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=self.bias,
            linear_method=linear_method,
        )
        scaling = self.head_size**-0.5
        rotary_dim = int(self.head_size * config.rotary_pct)
        assert rotary_dim % 2 == 0
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.rotary_emb = get_rope(
            self.head_size,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
        )
        self.attn = RadixAttention(self.num_heads, 
                                   self.head_size, 
                                   scaling, 
                                   num_kv_heads=self.num_heads,
                                   layer_id=layer_id)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v, input_metadata)
        output, _ = self.dense(attn_output)
        return output


class GPTNeoXMLP(nn.Module):

    def __init__(
        self,
        config: GPTNeoXConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            linear_method=linear_method,
        )
        self.dense_4h_to_h = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            linear_method=linear_method,
        )
        quant_config = getattr(linear_method, "quant_config", None)
        self.act = get_act_fn(config.hidden_act, quant_config,
                              config.intermediate_size)

    def forward(self, hidden_states):
        hidden_states, _ = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.dense_4h_to_h(hidden_states)
        return hidden_states


class GPTNeoXLayer(nn.Module):

    def __init__(
        self,
        config: GPTNeoXConfig,
        layer_id: int = 0,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                     eps=config.layer_norm_eps)
        self.attention = GPTNeoXAttention(config, layer_id=layer_id, linear_method=linear_method)
        self.mlp = GPTNeoXMLP(config, linear_method)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        attn_input = self.input_layernorm(hidden_states)
        attn_output = self.attention(
            position_ids=position_ids,
            hidden_states=attn_input,
            input_metadata=input_metadata,
        )

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_input = self.post_attention_layernorm(attn_output)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output + attn_output
        return hidden_states


class GPTNeoXModel(nn.Module):

    def __init__(
        self,
        config: GPTNeoXConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config

        self.embed_in = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            GPTNeoXLayer(config, i, linear_method)
            for i in range(config.num_hidden_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size,
                                             eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        input_metadata: InputMetadata,
        skip_embed: bool = False,
    ) -> torch.Tensor:
        if not skip_embed:
            hidden_states = self.embed_in(input_ids)
        else:
            hidden_states = input_ids
        hidden_states = self.embed_in(input_ids)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                position_ids,
                hidden_states,
                input_metadata,
            )
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class GPTNeoXForCausalLM(nn.Module):

    def __init__(
        self,
        config,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.gpt_neox = GPTNeoXModel(config, linear_method)
        self.embed_out = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        skip_embed: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.gpt_neox(input_ids, positions, input_metadata, skip_embed)
        return self.logits_processor(
            input_ids, hidden_states, self.embed_out.weight, input_metadata
        )

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if ("attention.bias" in name or "attention.masked_bias" in name
                    or "rotary_emb.inv_freq" in name):
                continue
            param = params_dict[name]

            if "query_key_value" in name:
                # NOTE: GPT-NeoX's fused QKV's output_dim has the shape of
                # (num_heads * 3 * head_size), while the
                # required shape is (3 * num_heads * head_size).
                # Thus, we need weight conversion.
                output_dim = getattr(param, "output_dim", None)
                num_heads = self.config.num_attention_heads
                if output_dim is not None:
                    loaded_weight_shape = loaded_weight.shape
                    loaded_weight = loaded_weight.view(
                        loaded_weight_shape[:output_dim] + (num_heads, 3, -1) +
                        loaded_weight_shape[output_dim + 1:])
                    loaded_weight = loaded_weight.transpose(
                        output_dim, output_dim + 1)
                    loaded_weight = loaded_weight.reshape(loaded_weight_shape)

            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

EntryClass = GPTNeoXForCausalLM