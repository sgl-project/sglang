# Copyright 2023-2024 SGLang Team
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

"""Kimi K2 model configuration - similar to DeepSeek V3 with MLA attention."""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class KimiK2Config(PretrainedConfig):
    """Configuration class for Kimi K2 models (moonshotai/Kimi-K2-*)."""

    model_type = "kimi_k2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 61,
        num_attention_heads: int = 128,
        num_key_value_heads: int = 128,
        n_shared_experts: int = 1,
        n_routed_experts: int = 256,
        num_experts_per_tok: int = 8,
        routed_scaling_factor: float = 2.5,
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # MLA (Multi-head Latent Attention) specific parameters
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 64,
        kv_lora_rank: int = 512,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        # MoE specific parameters
        first_k_dense_replace: int = 1,
        moe_layer_freq: int = 1,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        n_group: int = 8,
        topk_group: int = 4,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # For backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # MLA parameters
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim

        # MoE parameters
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self) -> bool:
        """Check if this model uses MLA (Multi-head Latent Attention)."""
        return self.kv_lora_rank is not None

    @property
    def is_moe(self) -> bool:
        """Check if this model uses MoE (Mixture of Experts)."""
        return self.n_routed_experts is not None and self.n_routed_experts > 0

