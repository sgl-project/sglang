# Copyright 2023-2026 SGLang Team
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
"""Inference-only JetBrains Mellum model compatible with HuggingFace weights."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_attn_context_model_parallel_rank,
    get_attn_context_model_parallel_world_size,
    get_moe_data_parallel_world_size,
    get_pp_group,
)
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding, get_rope
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix


def get_attention_sliding_window_size(config: PretrainedConfig) -> Optional[int]:
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None or sliding_window <= 0:
        return None
    # SGLang's RadixAttention window is exclusive; HF's config value is inclusive.
    return sliding_window - 1


def _get_layer_type(config: PretrainedConfig, layer_id: int) -> str:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return "full_attention"
    return layer_types[layer_id]


def _get_mlp_layer_type(config: PretrainedConfig, layer_id: int) -> str:
    mlp_layer_types = getattr(config, "mlp_layer_types", None)
    if mlp_layer_types is None:
        return "sparse"
    return mlp_layer_types[layer_id]


def _to_sglang_rope_scaling(
    rope_parameters: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not rope_parameters:
        return None

    rope_type = rope_parameters.get("rope_type") or rope_parameters.get("type")
    if rope_type in (None, "default"):
        return None

    rope_scaling: Dict[str, Any] = {"rope_type": rope_type}
    pass_through = (
        "factor",
        "original_max_position_embeddings",
        "beta_fast",
        "beta_slow",
        "extrapolation_factor",
        "truncate",
        "low_freq_factor",
        "high_freq_factor",
        "mscale",
        "mscale_all_dim",
        "short_factor",
        "long_factor",
        "short_mscale",
        "long_mscale",
    )
    for key in pass_through:
        if key in rope_parameters:
            rope_scaling[key] = rope_parameters[key]
    if "attention_factor" in rope_parameters:
        rope_scaling["attn_factor"] = rope_parameters["attention_factor"]
    return rope_scaling


def _get_layer_rope_config(
    config: PretrainedConfig, layer_type: str
) -> tuple[float, Optional[Dict[str, Any]], PretrainedConfig, float]:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and layer_type in rope_parameters:
        layer_rope_parameters = rope_parameters[layer_type] or {}
    elif isinstance(rope_parameters, dict):
        layer_rope_parameters = rope_parameters
    else:
        layer_rope_parameters = getattr(config, "rope_scaling", None) or {}

    rope_theta = layer_rope_parameters.get(
        "rope_theta", getattr(config, "rope_theta", 10000.0)
    )
    rope_scaling = _to_sglang_rope_scaling(layer_rope_parameters)
    partial_rotary_factor = layer_rope_parameters.get("partial_rotary_factor", 1.0)

    flat_config = copy.copy(config)
    flat_rope_parameters = dict(layer_rope_parameters)
    if "rope_type" not in flat_rope_parameters and "type" not in flat_rope_parameters:
        flat_rope_parameters["rope_type"] = "default"
    flat_rope_parameters.setdefault("rope_theta", rope_theta)
    flat_config.rope_parameters = flat_rope_parameters
    return rope_theta, rope_scaling, flat_config, partial_rotary_factor


class MellumMLP(Qwen3MoeMLP):
    def forward(
        self,
        x: torch.Tensor,
        forward_batch=None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        del forward_batch
        return super().forward(x, should_allreduce_fusion, use_reduce_scatter)


class MellumAttention(Qwen3MoeAttention):
    def __init__(
        self,
        *args,
        sliding_window_size: Optional[int] = None,
        partial_rotary_factor: float = 1.0,
        **kwargs,
    ) -> None:
        rope_theta = kwargs.get("rope_theta", 10000)
        rope_scaling = kwargs.get("rope_scaling", None)
        max_position_embeddings = kwargs.get("max_position_embeddings", 8192)
        super().__init__(*args, **kwargs)
        self.attn.sliding_window_size = sliding_window_size or -1

        if partial_rotary_factor < 1.0:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                partial_rotary_factor=partial_rotary_factor,
            )
            self.compatible_with_fused_kv_buffer = not isinstance(
                self.rotary_emb, MRotaryEmbedding
            )
            self.compatible_with_fused_qk_norm_rope = False
            self.use_fused_qk_norm_rope = False


class MellumDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        layer_type = _get_layer_type(config, layer_id)
        is_sliding = layer_type == "sliding_attention"
        rope_theta, rope_scaling, rope_config, partial_rotary_factor = (
            _get_layer_rope_config(config, layer_type)
        )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        self.self_attn = MellumAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            config=rope_config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            sliding_window_size=(
                get_attention_sliding_window_size(config) if is_sliding else None
            ),
            partial_rotary_factor=partial_rotary_factor,
        )

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        mlp_layer_type = _get_mlp_layer_type(config, layer_id)
        self.is_layer_sparse = mlp_layer_type == "sparse"
        is_previous_layer_sparse = (
            layer_id > 0 and _get_mlp_layer_type(config, layer_id - 1) == "sparse"
        )
        is_next_layer_sparse = (
            layer_id + 1 < config.num_hidden_layers
            and _get_mlp_layer_type(config, layer_id + 1) == "sparse"
        )

        if self.is_layer_sparse:
            self.mlp = Qwen3MoeSparseMoeBlock(
                layer_id=self.layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = MellumMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(self.layer_id == self.config.num_hidden_layers - 1),
        )


class MellumModel(Qwen3MoeModel):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=MellumDecoderLayer,
        )


class MellumForCausalLM(Qwen3MoeForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = MellumModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        self.attn_cp_size = get_attn_context_model_parallel_world_size()
        self.attn_cp_rank = get_attn_context_model_parallel_rank()
        self.moe_dp_size = get_moe_data_parallel_world_size()

        assert self.attn_cp_size % self.moe_dp_size == 0, (
            f"attn_cp_size ({self.attn_cp_size}) must be divisible by "
            f"moe_dp_size ({self.moe_dp_size})"
        )

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return get_attention_sliding_window_size(self.config)


EntryClass = MellumForCausalLM
