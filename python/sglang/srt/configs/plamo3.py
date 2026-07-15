# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PLaMo3 model configuration."""

from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


def is_full_attn(sliding_window_pattern: int, layer_idx: int) -> bool:
    return not bool((layer_idx + 1) % sliding_window_pattern)


class Plamo3Config(PretrainedConfig):  # type: ignore[misc]
    model_type: str = "plamo3"

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        # Attention
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        max_position_embeddings: int = 2048,
        window_size: int = 2048,
        sliding_window_pattern: int = 8,
        rope_theta: int = 1_000_000,
        rope_local_theta: int = 10_000,
        rope_scaling_factor: float = 1,
        initial_context_length: Optional[int] = None,
        attention_bias: bool = False,
        # MLP
        intermediate_size: int = 13312,
        hidden_activation: str = "swiglu",
        # Tokenizer
        vocab_size: int = 32000,
        tokenizer_class: str = "Plamo3Tokenizer",
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # Evaluation
        use_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.window_size = window_size
        self.sliding_window_pattern = sliding_window_pattern
        self.rope_theta = rope_theta
        self.rope_local_theta = rope_local_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.initial_context_length = initial_context_length
        self.attention_bias = attention_bias
        self.intermediate_size = intermediate_size
        self.hidden_activation = hidden_activation
        self.vocab_size = vocab_size
        self.use_cache = use_cache

        self.interleaved_sliding_window: list[int | None] = []
        for i in range(self.num_hidden_layers):
            if is_full_attn(self.sliding_window_pattern, i):
                self.interleaved_sliding_window.append(None)
            else:
                self.interleaved_sliding_window.append(self.window_size)
        assert len(self.interleaved_sliding_window) == self.num_hidden_layers

        if "architectures" not in kwargs:
            kwargs["architectures"] = ["Plamo3ForCausalLM"]

        super().__init__(
            tokenizer_class=tokenizer_class,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layer_types(self) -> list[str]:
        return [
            "full_attention" if sliding_window_size is None else "sliding_attention"
            for sliding_window_size in self.interleaved_sliding_window
        ]

    @property
    def layers_block_type(self) -> list[str]:
        return ["attention" for _ in range(self.num_hidden_layers)]

    @property
    def rope_scaling(self) -> dict[str, Any] | None:
        if self.rope_scaling_factor == 1:
            return None
        assert self.initial_context_length is not None
        return {
            "full_attention": {
                "rope_theta": self.rope_theta,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": self.rope_scaling_factor,
                "original_max_position_embeddings": self.initial_context_length,
                "rope_type": "yarn",
                "truncate": False,
            },
            "sliding_attention": {
                "rope_theta": self.rope_local_theta,
                "rope_type": "default",
            },
        }

    @property
    def rope_local_base_freq(self) -> int:
        return self.rope_local_theta
