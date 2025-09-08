# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
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
"""Qwen3Hybrid model configuration"""

import enum

import numpy as np
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

from sglang.srt.distributed.utils import divide
from sglang.srt.layers.dp_attention import get_attention_tp_size

logger = logging.get_logger(__name__)


# NOTE: HybridLayerType
class HybridLayerType(enum.Enum):
    full_attention = "attention"
    swa_attention = "swa_attention"
    linear_attention = "linear_attention"
    mamba2 = "mamba"


class Qwen3NextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3HybridModel`]. It is used to instantiate a
    Qwen3Hybrid model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
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
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        hybrid (`bool`, *optional*, defaults to `True`):
            Whether is a hybrid architecture.
        hybrid_full_attention (`bool`, *optional*, defaults to `True`):
            Whether to use full attention layer in hybrid model.
        hybrid_mamba2 (`bool`, *optional*, defaults to `True`):
            Whether to use mamba2 layer in hybrid model.
        full_attention_interval (`int`, *optional*, defaults to `1`):
            Full attention layer interval.
        mamba2_interval (`int`, *optional*, defaults to `1`):
            Mamba2 layer interval.
        mamba2_state_dim (`int`, *optional*, defaults to 128):
            The dimension the mamba state space latents
        mamba2_ngroups (`int`, *optional*, defaults to 8):
            The number of the mamba groups used in the v2 implementation.
        mamba2_head_dim (`int`, *optional*, defaults to 128):
            The dimension of mamba heads used in the v2 implementation.
        mamba2_nheads (`int`, *optional*, defaults to `"auto"`):
            The number of mamba heads used in the v2 implementation.
        mamba2_conv_dim (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel.
        mamba2_expand (`int`, *optional*, defaults to 1):
            Expanding factor (relative to hidden_size) used to determine the mamba2 intermediate size
        mamba2_chunk_size (`int`, *optional*, defaults to 128):
            The chunks in which to break the sequence when doing prefill/training. We use the value same in Megatron code. Need to check for better speed?
        mamba2_conv_bias (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use bias in the convolution layer of the mamba2 mixer block.
        mamba2_proj_bias (`bool`, *optional*, defaults to `False`):
            Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the mamba mixer block

    ```python
    >>> from transformers import Qwen3HybridModel, Qwen3HybridConfig

    >>> # Initializing a Qwen3 style configuration
    >>> configuration = Qwen3HybridConfig()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen3HybridModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_next"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        use_qk_norm=False,
        head_dim=None,
        qkv_bias=False,
        hybrid=True,
        hybrid_full_attention=True,
        hybrid_mamba2=False,
        hybrid_linear_attention=False,
        full_attention_interval=1,
        mamba2_interval=1,
        linear_attention_interval=1,
        mamba2_state_dim=128,
        mamba2_ngroups=8,
        mamba2_head_dim=128,
        mamba2_nheads=32,
        mamba2_conv_dim=4,
        mamba2_expand=1,
        mamba2_chunk_size=128,
        mamba2_conv_bias=True,
        mamba2_proj_bias=False,
        rescale_prenorm_residual=True,
        position_embedding_type=None,
        linear_attention_type="gated_deltanet",
        linear_expand_v=1,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=8,
        linear_num_value_heads=16,
        norm_before_gate=True,
        output_gate_type="swish",
        share_norm=True,
        output_norm_size="head",
        decoder_sparse_step=1,
        moe_intermediate_size=768,
        shared_expert_intermediate_size=0,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

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
        self.use_qk_norm = use_qk_norm
        self.head_dim = head_dim
        self.qkv_bias = qkv_bias
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # for hybrid part
        self.hybrid = hybrid
        self.hybrid_full_attention = hybrid_full_attention
        self.hybrid_mamba2 = hybrid_mamba2
        self.hybrid_linear_attention = hybrid_linear_attention
        self.full_attention_interval = full_attention_interval
        self.mamba2_interval = mamba2_interval
        self.linear_attention_interval = linear_attention_interval

        # mamba2 part
        self.mamba2_expand = mamba2_expand
        self.mamba2_d_inner = mamba2_expand * self.hidden_size
        if self.mamba2_d_inner % mamba2_head_dim != 0:
            raise ValueError("mamba2_head_dim must divide mamba2_expand * hidden_size")

        self.mamba2_head_dim = mamba2_head_dim
        self.mamba2_nheads = mamba2_nheads
        if self.mamba2_head_dim * self.mamba2_nheads != self.mamba2_d_inner:
            raise ValueError(
                "The dimensions for the Mamba2 head state do not match the model mamba2 d_inner"
            )

        self.mamba2_ngroups = mamba2_ngroups
        self.mamba2_state_dim = mamba2_state_dim
        self.mamba2_conv_dim = mamba2_conv_dim
        self.mamba2_chunk_size = mamba2_chunk_size
        self.mamba2_conv_bias = mamba2_conv_bias
        self.mamba2_proj_bias = mamba2_proj_bias

        # linear attention (gdn now part)
        self.linear_attention_type = linear_attention_type
        self.linear_expand_v = linear_expand_v
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.norm_before_gate = norm_before_gate
        self.output_gate_type = output_gate_type
        self.share_norm = share_norm
        self.output_norm_size = output_norm_size

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

        # NOTE: this is same as in megatron code. If megatron change, need to change here too.
        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4

        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.position_embedding_type = position_embedding_type
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        layer_type_list = []

        for l in range(self.num_hidden_layers):
            if (
                l + 1
            ) % self.full_attention_interval == 0 and self.hybrid_full_attention:
                layer_type_list.append(HybridLayerType.full_attention.value)
            elif (
                l + 1
            ) % self.linear_attention_interval == 0 and self.hybrid_linear_attention:
                layer_type_list.append(HybridLayerType.linear_attention.value)
            elif (l + 1) % self.mamba2_interval == 0 and self.hybrid_mamba2:
                layer_type_list.append(HybridLayerType.mamba2.value)

        return layer_type_list

    @property
    def linear_layer_ids(self):
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == HybridLayerType.linear_attention.value
        ]

    @property
    def full_attention_layer_ids(self):
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == HybridLayerType.full_attention.value
        ]

    @property
    def hybrid_gdn_params(self):
        world_size = get_attention_tp_size()
        conv_dim = (
            self.linear_key_head_dim * self.linear_num_key_heads * 2
            + self.linear_value_head_dim * self.linear_num_value_heads
        )
        conv_state_shape = (
            divide(conv_dim, world_size),
            self.linear_conv_kernel_dim - 1,
        )

        temporal_state_shape = (
            divide(self.linear_num_value_heads, world_size),
            self.linear_key_head_dim,
            self.linear_value_head_dim,
        )
        conv_dtype = torch.bfloat16
        ssm_dtype = torch.float32
        mamba_layers = self.linear_layer_ids
        return (
            conv_state_shape,
            temporal_state_shape,
            conv_dtype,
            ssm_dtype,
            mamba_layers,
        )

    @property
    def mamba_cache_per_req(self):
        conv_state_shape, temporal_state_shape, conv_dtype, ssm_dtype, mamba_layers = (
            self.hybrid_gdn_params
        )
        mamba_layers_len = len(mamba_layers)

        return (
            int(np.prod(conv_state_shape)) * conv_dtype.itemsize
            + int(np.prod(temporal_state_shape)) * ssm_dtype.itemsize
        ) * mamba_layers_len
