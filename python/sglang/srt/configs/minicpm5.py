# Copyright 2025-2026 SGLang Team
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
"""Native config for the MiniCPM5 MoE model (model_type="minicpm5_moe")."""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class MiniCPM5Config(PretrainedConfig):
    """Configuration with the MiniCPM5-15B-A2.5B defaults."""

    model_type = "minicpm5_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 130560,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = 2,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: Optional[int] = 1,
        bos_token_id: Optional[int] = 0,
        eos_token_id: int | list[int] | tuple[int, ...] | None = (1, 130073),
        tie_word_embeddings: bool = False,
        rope_theta: float = 5000000.0,
        rope_scaling: Optional[dict] = None,
        rope_parameters: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        first_k_dense_replace: int = 1,
        n_routed_experts: int = 160,
        n_shared_experts: int = 1,
        routed_scaling_factor: float = 3.66,
        topk_method: str = "greedy",
        num_experts_per_tok: Optional[int] = 16,
        moe_intermediate_size: int = 512,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        use_gated_attention: bool = True,
        *,
        architectures: list[str] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        if head_dim is None or head_dim <= 0:
            raise ValueError("MiniCPM5 head_dim must be positive.")
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        parameters = dict(
            (rope_scaling or {}) if rope_parameters is None else rope_parameters
        )
        legacy_type = parameters.pop("type", "default")
        parameters.setdefault("rope_type", legacy_type)
        parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters = parameters
        self.rope_theta = parameters["rope_theta"]
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.first_k_dense_replace = first_k_dense_replace
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.use_gated_attention = use_gated_attention

        if architectures == ["MiniCPM5MoEForCausalLM"]:
            architectures = ["MiniCPM5ForCausalLM"]

        if isinstance(eos_token_id, tuple):
            eos_token_id = list(eos_token_id)
        super().__init__(
            architectures=architectures,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
