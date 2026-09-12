# SPDX-License-Identifier: Apache-2.0
"""Cohere2Moe text config used by the Cohere Command-A Plus checkpoints."""

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


class Cohere2MoeConfig(PreTrainedConfig):
    model_type = "cohere2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_size: int = 8192,
        intermediate_size: int = 22528,
        logit_scale: float = 0.0625,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 5,
        eos_token_id: int | list[int] | None = 255001,
        tie_word_embeddings: bool = True,
        rope_theta: float | int = 10000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int | None = 4096,
        num_experts_per_tok: int = 2,
        num_experts: int = 8,
        num_shared_experts: int = 0,
        shared_expert_combination_strategy: str = "average",
        expert_selection_fn: str = "softmax",
        layer_types: list[str] | None = None,
        first_k_dense_replace: int = 0,
        prefix_dense_sliding_window_pattern: int = 1,
        norm_topk_prob: bool = True,
        prefix_dense_intermediate_size: int | None = None,
        rms_norm_eps: float | None = None,
        sliding_window_pattern: int = 4,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.logit_scale = logit_scale
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.shared_expert_combination_strategy = shared_expert_combination_strategy
        self.expert_selection_fn = expert_selection_fn
        self.first_k_dense_replace = first_k_dense_replace
        self.prefix_dense_sliding_window_pattern = prefix_dense_sliding_window_pattern
        self.norm_topk_prob = norm_topk_prob
        self.prefix_dense_intermediate_size = prefix_dense_intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window_pattern = sliding_window_pattern

        if layer_types is None:
            prefix_layers = [
                (
                    "sliding_attention"
                    if ((i + 1) % prefix_dense_sliding_window_pattern) != 0
                    else "full_attention"
                )
                for i in range(self.first_k_dense_replace)
            ]
            rest_layers = [
                (
                    "sliding_attention"
                    if ((i + 1) % sliding_window_pattern) != 0
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers - self.first_k_dense_replace)
            ]
            self.layer_types = prefix_layers + rest_layers
        else:
            self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )

        if hasattr(self, "standardize_rope_params"):
            try:
                self.standardize_rope_params()
                self.validate_rope()
            except Exception:
                pass


try:
    CONFIG_MAPPING.register("cohere2_moe", Cohere2MoeConfig)
except Exception:
    CONFIG_MAPPING._extra_content["cohere2_moe"] = Cohere2MoeConfig
