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

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/telechat2.py
"""Inference-only LLaMA model compatible with HuggingFace weights."""



import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import LlamaConfig

from sglang.srt.models.llama import LlamaForCausalLM, LlamaModel
from .llama import LlamaDecoderLayer
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
    maybe_remap_kv_scale_name,
)

logger = logging.getLogger(__name__)


class TeleChat2Model(LlamaModel):
    def __init__(self, config: LlamaConfig, prefix: str = ""):
        # 1. Initialize the LlamaModel with bias
        config.attention_bias = True
        config.mlp_bias = True    
        super().__init__(config=config, prefix=prefix)
        # 2. Remove the bias from the qkv_proj and gate_up_proj based on config
        # Telechat2's gate_up_proj and qkv_proj don't have bias
        # see: https://github.com/vllm-project/vllm/pull/10311#issuecomment-2490297566
        for layer in self.layers:
            if not isinstance(layer, PPMissingLayer):
                layer.self_attn.qkv_proj.bias = None
                layer.self_attn.qkv_proj.skip_bias_add = True
                layer.mlp.gate_up_proj.bias = None
                layer.mlp.gate_up_proj.skip_bias_add = True


class TeleChat2ForCausalLM(LlamaForCausalLM):
    
    # Map to change the hf naming to sg lang style
    prefix_map = {
        "transformer.": "model.",
    }

    substr_map = {
        ".h.": ".layers.",
        ".self_attention.": ".self_attn.",
        ".word_embeddings.": ".embed_tokens.",
        ".dense.": ".o_proj.",
        ".ln_f.": ".norm.",
    }

    # Apply mappings
    @staticmethod
    def remap_telechat2_name(name):
        for old_prefix, new_prefix in TeleChat2ForCausalLM.prefix_map.items():
            if name.startswith(old_prefix):
                name = name.replace(old_prefix, new_prefix, 1)
                break

        for old_substr, new_substr in TeleChat2ForCausalLM.substr_map.items():
            name = name.replace(old_substr, new_substr)

        return name

    @staticmethod
    def patch_config(config):
        if not hasattr(config, "intermediate_size") and hasattr(config, "ffn_hidden_size"):
            config.intermediate_size = config.ffn_hidden_size

        if not hasattr(config, "hidden_act") and hasattr(config, "activation_function"):
            config.hidden_act = config.activation_function
        elif not hasattr(config, "hidden_act"):
            config.hidden_act = "silu"

        if not hasattr(config, "rms_norm_eps") and hasattr(config, "layer_norm_epsilon"):
            config.rms_norm_eps = config.layer_norm_epsilon
        elif not hasattr(config, "rms_norm_eps"):
            config.rms_norm_eps = 1e-6

        if not hasattr(config, "initializer_range") and hasattr(config, "init_std"):
            config.initializer_range = config.init_std
        elif not hasattr(config, "initializer_range"):
            config.initializer_range = 0.02

        if not hasattr(config, "num_key_value_heads") and hasattr(config, "num_attention_heads"):
            config.num_key_value_heads = config.num_attention_heads

        if not hasattr(config, "rope_theta"):
            config.rope_theta = 10000.0

        if not hasattr(config, "attention_bias"):
            config.attention_bias = False

        if not hasattr(config, "attention_dropout"):
            config.attention_dropout = 0.0

        if not hasattr(config, "mlp_bias"):
            config.mlp_bias = False

        if not hasattr(config, "head_dim") and hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            config.head_dim = config.hidden_size // config.num_attention_heads


    def _init_model(self,
                    config: LlamaConfig,
                    prefix: str = "",
                    layer_type: type[nn.Module] = LlamaDecoderLayer):
        TeleChat2ForCausalLM.patch_config(config)
        return TeleChat2Model(config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ('gate_up_proj', 'gate_proj', 0),
            ('gate_up_proj', 'up_proj', 1),
        ]
        params_dict = dict(self.named_parameters())
        total_num_heads = self.config.n_head
        head_dim = self.config.hidden_size // total_num_heads
        for name, loaded_weight in weights:
            name = TeleChat2ForCausalLM.remap_telechat2_name(name)
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "self_attn.key_value" in name:
                k_weight = []
                v_weight = []
                for i in range(total_num_heads):
                    start = i * head_dim * 2
                    k_weight.append(loaded_weight[start:start + head_dim, :])
                    v_weight.append(loaded_weight[start + head_dim:start +
                                                  2 * head_dim:])
                k_weight = torch.cat(k_weight, dim=0)
                v_weight = torch.cat(v_weight, dim=0)
                name = name.replace("key_value", "qkv_proj")
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, k_weight, "k")
                weight_loader(param, v_weight, "v")
            elif "query" in name:
                name = name.replace("query", "qkv_proj")
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, "q")
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

EntryClass = [TeleChat2ForCausalLM]