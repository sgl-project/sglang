# SPDX-License-Identifier: Apache-2.0
"""Cohere2Moe text config used by the Cohere Command-A Plus checkpoints."""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


class Cohere2MoeConfig(PretrainedConfig):
    model_type = "cohere2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 256000
    hidden_size: int = 8192
    intermediate_size: int = 22528
    logit_scale: float = 0.0625
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int | None = None
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 5
    eos_token_id: int | list[int] | None = 255001
    tie_word_embeddings: bool = True
    rope_theta: float | int = 10000.0
    rope_scaling: dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int | None = 4096
    num_experts_per_tok: int = 2
    num_experts: int = 8
    num_shared_experts: int = 0
    shared_expert_combination_strategy: str = "average"
    expert_selection_fn: str = "softmax"
    layer_types: list[str] | None = None
    first_k_dense_replace: int = 0
    prefix_dense_sliding_window_pattern: int = 1
    norm_topk_prob: bool = True
    prefix_dense_intermediate_size: int | None = None
    rms_norm_eps: float | None = None
    sliding_window_pattern: int = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__post_init__(**kwargs)

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if hasattr(self, "standardize_rope_params"):
            try:
                self.standardize_rope_params()
                self.validate_rope()
            except Exception:
                pass

        if self.layer_types is None:
            prefix_layers = [
                (
                    "sliding_attention"
                    if ((i + 1) % self.prefix_dense_sliding_window_pattern) != 0
                    else "full_attention"
                )
                for i in range(self.first_k_dense_replace)
            ]
            rest_layers = [
                (
                    "sliding_attention"
                    if ((i + 1) % self.sliding_window_pattern) != 0
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers - self.first_k_dense_replace)
            ]
            self.layer_types = prefix_layers + rest_layers


try:
    CONFIG_MAPPING.register("cohere2_moe", Cohere2MoeConfig)
except Exception:
    CONFIG_MAPPING._extra_content["cohere2_moe"] = Cohere2MoeConfig
