# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3-VL encoder configuration for MiniMax H3."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.qwen3vl import (
    Qwen3VLArchConfig,
    Qwen3VLConfig,
)

MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER = 50


@dataclass
class MiniMaxH3Qwen3VLArchConfig(Qwen3VLArchConfig):
    """The checkpoint is Qwen3-VL-32B, consumed at hidden_states[50]."""

    architectures: list[str] = field(
        default_factory=lambda: ["MiniMaxH3Qwen3VLEncoder"]
    )
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_hidden_layers: int = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    text_len: int = 262144
    hidden_state_skip_layer: int = 0


@dataclass
class MiniMaxH3Qwen3VLConfig(Qwen3VLConfig):
    arch_config: MiniMaxH3Qwen3VLArchConfig = field(
        default_factory=MiniMaxH3Qwen3VLArchConfig
    )

    def post_diffusers_config_update(self) -> None:
        """Select the in-tree extractor after loading the HF architecture."""

        arch = self.arch_config
        arch.architectures = ["MiniMaxH3Qwen3VLEncoder"]
        arch.hidden_size = int(arch.text_config.hidden_size)
        arch.intermediate_size = int(arch.text_config.intermediate_size)
        arch.num_attention_heads = int(arch.text_config.num_attention_heads)
        arch.num_key_value_heads = int(arch.text_config.num_key_value_heads)
        arch.head_dim = int(arch.text_config.head_dim)
        arch.num_hidden_layers = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        arch.text_config.num_hidden_layers = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        arch.text_config.output_hidden_states = False
        arch.text_config.use_cache = False


__all__ = [
    "MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER",
    "MiniMaxH3Qwen3VLArchConfig",
    "MiniMaxH3Qwen3VLConfig",
]
