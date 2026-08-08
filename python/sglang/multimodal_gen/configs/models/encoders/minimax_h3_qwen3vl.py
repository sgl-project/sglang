# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3-VL encoder configuration for MiniMax H3."""

import logging
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.models.encoders.qwen3vl import (
    Qwen3VLArchConfig,
    Qwen3VLConfig,
)

logger = logging.getLogger(__name__)

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
        self._resolve_quant_config()

    def _resolve_quant_config(self) -> None:
        """Adopt the quantization scheme a pre-quantized checkpoint ships, if any."""

        source = getattr(self.arch_config, "quantization_config", None)
        if source is None:
            self.quant_config = None
            return
        config_dict: dict[str, Any] = (
            dict(source) if isinstance(source, dict) else source.to_dict()
        )
        quant_method = str(config_dict.get("quant_method", "")).lower()
        if quant_method != "fp8":
            raise ValueError(
                "MiniMax H3 Qwen3-VL text encoder only supports 'fp8' quantized "
                f"checkpoints; got quant_method={quant_method!r}. Point "
                "--text-encoder-path at a BF16 or FP8 Qwen3-VL release."
            )

        from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config

        self.quant_config = Fp8Config.from_config(config_dict)
        logger.info(
            "MiniMax H3 text encoder: loading FP8 checkpoint "
            "(weight_block_size=%s, activation_scheme=%s)",
            self.quant_config.weight_block_size,
            self.quant_config.activation_scheme,
        )


__all__ = [
    "MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER",
    "MiniMaxH3Qwen3VLArchConfig",
    "MiniMaxH3Qwen3VLConfig",
]
