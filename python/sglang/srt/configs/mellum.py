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
"""JetBrains Mellum model configuration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers.configuration_utils import PretrainedConfig


class MellumConfig(PretrainedConfig):
    model_type = "mellum"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 98304,
        hidden_size: int = 2304,
        intermediate_size: int = 7168,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: Optional[Dict[str, Any]] = None,
        attention_bias: bool = False,
        sliding_window: Optional[int] = 1024,
        attention_dropout: float = 0.0,
        moe_intermediate_size: int = 896,
        num_experts_per_tok: int = 8,
        num_experts: int = 64,
        norm_topk_prob: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        head_dim: int = 128,
        layer_types: Optional[List[str]] = None,
        mlp_layer_types: Optional[List[str]] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[Any] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.head_dim = head_dim

        self.layer_types = (
            list(layer_types)
            if layer_types is not None
            else ["full_attention"] * num_hidden_layers
        )
        self.mlp_layer_types = (
            list(mlp_layer_types)
            if mlp_layer_types is not None
            else ["sparse"] * num_hidden_layers
        )

        if rope_parameters is None:
            rope_parameters = {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 500000.0,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
            }
        self.rope_parameters = rope_parameters
        if isinstance(rope_parameters, dict):
            self.rope_theta = rope_parameters.get("full_attention", {}).get(
                "rope_theta", rope_parameters.get("rope_theta", 500000.0)
            )
        else:
            self.rope_theta = 500000.0

        self.is_hybrid_swa = "sliding_attention" in self.layer_types
        if getattr(self, "architectures", None) is None:
            self.architectures = ["MellumForCausalLM"]
