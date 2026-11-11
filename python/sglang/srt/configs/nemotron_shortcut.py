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
"""Nemotron Shortcut's hybrid layout, read from the checkpoint's own config.

The checkpoint ships its config and processor as remote code, loaded with
``--trust-remote-code``. Its normalized ``layer_types`` name each block:

    linear_attention   GDP mixer
    mlp                dense MLP
    full_attention     attention mixer
    shortcut_moe_gdp   GDP mixer + shortcut MoE
    shortcut_moe_attn  attention mixer + shortcut MoE

The shortcut MoE's routed experts read the block input, so dispatch can start
before the mixer runs.
"""

from __future__ import annotations

from typing import Optional

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)

# The checkpoint's text-config model_type.
TEXT_MODEL_TYPE = "nemotron4_text"

GDP_LAYER_TYPES = ("linear_attention", "shortcut_moe_gdp")
ATTENTION_LAYER_TYPES = ("full_attention", "shortcut_moe_attn")
MOE_LAYER_TYPES = ("shortcut_moe_gdp", "shortcut_moe_attn")


def is_nemotron_shortcut_text_config(config: PretrainedConfig) -> bool:
    return config.model_type == TEXT_MODEL_TYPE


def gdp_value_dim(config: PretrainedConfig) -> int:
    return config.gdp_num_heads * config.gdp_head_dim


def gdp_key_dim(config: PretrainedConfig) -> int:
    return config.gdp_num_groups * config.gdp_key_head_dim


def gdp_conv_dim(config: PretrainedConfig) -> int:
    """Conv channels: n value blocks, then n key blocks and the query."""
    n = config.gdp_num_householder
    return n * gdp_value_dim(config) + (n + 1) * gdp_key_dim(config)


class NemotronShortcutLayout:
    """The hybrid bookkeeping SGLang's state and KV pools read, which the
    checkpoint's config does not carry."""

    def __init__(self, text_config: PretrainedConfig):
        if text_config.gdp_num_heads != text_config.gdp_num_groups:
            # FlashInfer's GDP kernels take one head count for q, k and v.
            raise ValueError(
                "Nemotron Shortcut GDP requires gdp_num_groups == gdp_num_heads, "
                f"got {text_config.gdp_num_groups} vs {text_config.gdp_num_heads}."
            )
        self.text_config = text_config
        layer_types = text_config.layer_types
        self.linear_layer_ids = [
            i for i, t in enumerate(layer_types) if t in GDP_LAYER_TYPES
        ]
        self.full_attention_layer_ids = [
            i for i, t in enumerate(layer_types) if t in ATTENTION_LAYER_TYPES
        ]

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        """Per-request GDP state: conv window over [V0..Vn, K0..Kn, Q] and one
        FP32 [H, V, K] recurrent state per GDP layer."""
        from sglang.srt.runtime_context import get_parallel

        config = self.text_config
        n = config.gdp_num_householder
        value_dim, key_dim = gdp_value_dim(config), gdp_key_dim(config)
        # Mamba2StateShape's conv is intermediate + 2 * n_groups * state; GDP's is
        # n value blocks plus n + 1 key-sized blocks (n keys and the query).
        assert ((n + 1) * config.gdp_num_groups) % 2 == 0
        shape = Mamba2StateShape.create(
            tp_world_size=get_parallel().attn_tp_size,
            intermediate_size=n * value_dim,
            n_groups=(n + 1) * config.gdp_num_groups // 2,
            num_heads=config.gdp_num_heads,
            head_dim=config.gdp_head_dim,
            state_size=config.gdp_key_head_dim,
            conv_kernel=config.gdp_conv_kernel,
            conv_shard_groups=[value_dim] * n + [key_dim] * (n + 1),
        )
        assert shape.conv_dim == gdp_conv_dim(config), (
            shape.conv_dim,
            gdp_conv_dim(config),
        )
        return Mamba2CacheParams(
            shape=shape,
            layers=self.linear_layer_ids,
            dtype=mamba2_state_dtype(config),
        )


def sdsa_index_head_dim(hf_config: PretrainedConfig) -> Optional[int]:
    """Index-key width when SDSA serves this model's attention, else None."""
    text_config = hf_config.get_text_config()
    if is_nemotron_shortcut_text_config(text_config) and text_config.index_topk:
        return text_config.index_head_dim
    return None


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=PretrainedConfig,
        config_predicate=is_nemotron_shortcut_text_config,
        config_view=NemotronShortcutLayout,
        backend_class_name=(
            "sglang.srt.layers.attention.linear.gdp_backend.GDPAttnBackend"
        ),
        arch_names=["Nemotron4ForConditionalGeneration"],
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
        support_mamba_cache_extra_buffer=True,
        unwrap_text_config=True,
        full_attn_backend_factory_name=(
            "sglang.srt.layers.attention.sdsa_backend.create_sdsa_backend"
        ),
    )
)
