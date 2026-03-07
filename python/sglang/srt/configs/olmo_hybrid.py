# coding=utf-8
# Copyright 2026 SGLang Team
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
"""OLMo-Hybrid model configuration."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)
from sglang.srt.configs.update_config import adjust_tp_num_heads_if_necessary
from sglang.srt.utils import is_cpu

logger = logging.get_logger(__name__)
_is_cpu = is_cpu()


class OlmoHybridConfig(PretrainedConfig):
    model_type = "olmo_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=3840,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=30,
        num_key_value_heads=30,
        hidden_act="silu",
        max_position_embeddings=65536,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        layer_types=None,
        linear_num_key_heads=30,
        linear_num_value_heads=30,
        linear_key_head_dim=96,
        linear_value_head_dim=192,
        linear_conv_kernel_dim=4,
        linear_allow_neg_eigval=True,
        **kwargs,
    ):
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["OlmoHybridForCausalLM"]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_allow_neg_eigval = linear_allow_neg_eigval

        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                "layer_types length must equal num_hidden_layers: "
                f"{len(self.layer_types)} != {self.num_hidden_layers}"
            )

    @property
    def layers_block_type(self) -> list[str]:
        return self.layer_types

    @property
    def linear_layer_ids(self) -> list[int]:
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == "linear_attention"
        ]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == "full_attention"
        ]

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        if _is_cpu:
            world_size = get_attention_tp_size()
            adjust_tp_num_heads_if_necessary(self, world_size, False)

        shape = Mamba2StateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=self.linear_value_head_dim * self.linear_num_value_heads,
            n_groups=self.linear_num_key_heads,
            num_heads=self.linear_num_value_heads,
            head_dim=self.linear_value_head_dim,
            state_size=self.linear_key_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
        )
        return Mamba2CacheParams(
            shape=shape, layers=self.linear_layer_ids, dtype=mamba2_state_dtype(self)
        )
