# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/configuration_qwen2.py

"""Jet-Nemotron model configuration"""

from dataclasses import dataclass

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    extra_groups_for_head_shards,
)
from sglang.srt.distributed.utils import divide
from sglang.srt.layers.dp_attention import get_attention_tp_size

logger = logging.get_logger(__name__)


class JetNemotronConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`JetModel`]. It is used to instantiate a
    Jet model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Jet-Nemotron model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`JetModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import JetModel, JetConfig

    >>> # Initializing a Jet-Nemotron style configuration
    >>> configuration = JetConfig()

    >>> # Initializing a model from the Jet-Nemotron-2B style configuration
    >>> model = Jet-Nemotron-Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "jet_nemotron"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        layer_types=None,
        efficient_attention_config=None,
        draft_vocab_size=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        if layer_types is None:
            self.layer_types = ["attn"] * num_hidden_layers
        elif isinstance(layer_types, str):
            self.layer_types = eval(layer_types)
        else:
            self.layer_types = layer_types
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.efficient_attention_config = efficient_attention_config
        self.draft_vocab_size = draft_vocab_size

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def linear_layer_ids(self):
        return [
            i for i in range(self.num_hidden_layers) if self.layer_types[i] == "jet"
        ]

    @property
    def full_attention_layer_ids(self):
        return [
            i
            for i in range(self.num_hidden_layers)
            if self.layer_types[i] == "attn" or self.layer_types[i] == "swa"
        ]

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        num_heads = self.efficient_attention_config["jet"]["num_heads"]
        value_head_dim = int(
            self.efficient_attention_config["jet"]["head_dim"]
            * self.efficient_attention_config["jet"]["expand_v"]
        )
        shape = JetNemotronStateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=value_head_dim * num_heads,
            n_groups=num_heads,
            num_heads=num_heads,
            head_dim=self.efficient_attention_config["jet"]["head_dim"],
            state_size=value_head_dim,
            conv_kernel=self.efficient_attention_config["jet"]["conv_size"],
        )

        return Mamba2CacheParams(shape=shape, layers=self.linear_layer_ids)


@dataclass(kw_only=True, frozen=True)
class JetNemotronStateShape(Mamba2StateShape):
    @staticmethod
    def create(
        *,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
    ) -> "JetNemotronStateShape":
        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        if n_groups % tp_world_size != 0:
            extra_groups = extra_groups_for_head_shards(n_groups, tp_world_size)
            n_groups += extra_groups
        # heads and n_groups are TP-ed
        conv_dim = intermediate_size

        # contiguous along 'dim' axis
        # conv_kernel not conv_kernel - 1 because our existing kernels are compatible with this, and also for alignment
        conv_state_shape = divide(conv_dim, tp_world_size), conv_kernel

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., QWen3-Next: (32, 128, 128)
        temporal_state_shape = (divide(num_heads, tp_world_size), head_dim, state_size)
        return JetNemotronStateShape(
            conv=conv_state_shape,
            temporal=temporal_state_shape,
            intermediate_size=intermediate_size,
            conv_dim=conv_dim,
            ssm_state_size=state_size,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
            conv_kernel=conv_kernel,
        )


@dataclass
class JetBlockConfig:
    mode: str = "chunk"
    expand_v: int = 2.0
    num_heads: int = 6
    head_dim: int = 256
    norm_eps: float = 1e-5
    conv_size: int = 4
    dconv_generator_reduction: int = 8
    dconv_implementation: str = "triton"
