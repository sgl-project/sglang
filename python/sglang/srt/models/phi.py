from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.router.model_runner import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn

from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)


class PhiAttention(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None,
                 layer_id: int = 0):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads

        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)

        # pylint: disable=C0103
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_size,
            self.total_num_heads,
            bias=True,
            linear_method=linear_method,
        )
        self.dense = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            linear_method=linear_method,
        )

        scaling = self.head_size**-0.5
        rotary_dim = int(config.partial_rotary_factor *
                         (config.hidden_size // config.num_attention_heads))
        assert rotary_dim % 2 == 0

        # pylint: disable=C0301
        # Refer to:
        # https://huggingface.co/microsoft/phi-1_5/blob/d212a789620c380ff32ca1d1ee9943a777360987/modeling_phi.py#L518
        rope_theta = 10000
        max_position_embeddings = getattr(config, "n_positions", 2048)
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
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v, input_metadata)
        output, _ = self.dense(attn_output)
        return output


class PhiMLP(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()

        n_inner = getattr(config, "n_inner", None)
        n_inner = n_inner if n_inner is not None else 4 * config.hidden_size

        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            n_inner,
            linear_method=linear_method,
        )
        self.fc2 = RowParallelLinear(
            n_inner,
            config.hidden_size,
            linear_method=linear_method,
        )
        quant_config = getattr(linear_method, "quant_config", None)
        self.act = get_act_fn(config.hidden_act, quant_config, n_inner)

    def forward(self, hidden_states):
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class PhiLayer(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None,
                 layer_id: int = 0):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)
        self.self_attn = PhiAttention(config, linear_method,layer_id=layer_id)
        self.mlp = PhiMLP(config, linear_method)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            input_metadata=input_metadata,
        )
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        return hidden_states


class PhiModel(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size)
        self.layers = nn.ModuleList([
            PhiLayer(config, linear_method, layer_id=i)
            for i in range(config.num_hidden_layers)
        ])
        self.final_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(self.config.num_hidden_layers):
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                input_metadata,
            )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class PhiForCausalLM(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.linear_method = linear_method

        self.model = PhiModel(config, linear_method)

        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      bias=True)
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions,
                                   input_metadata)
        
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head.weight, input_metadata
        )

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v")
        ]
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
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
                    continue
                # pylint: disable=E1136

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)