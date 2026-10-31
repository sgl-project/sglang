# SPDX-License-Identifier: Apache-2.0
"""AliceAI checkpoint configuration and SGLang's KDA cache contract."""

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    mamba2_state_dtype,
)
from sglang.srt.runtime_context import get_parallel


class AliceAIConfig(PretrainedConfig):
    model_type = "alice_ai"
    attribute_map = {"num_nextn_predict_layers": "mtp_num_hidden_layers"}
    keys_to_ignore_at_inference = ("past_key_values",)

    def __init__(
        self,
        vocab_size=129024,
        hidden_size=2048,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=32,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        block_attn_res_block_size=4,
        router_score_function="sigmoid",
        router_bias_correction=True,
        norm_topk_prob=True,
        kda_allow_negative_eigenvalues=False,
        max_position_embeddings=262144,
        rope_theta=1_000_000.0,
        rope_scaling=None,
        rope_parameters=None,
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        initializer_range=0.02,
        attention_dropout=0.0,
        use_cache=True,
        output_router_logits=False,
        layer_types=None,
        tie_word_embeddings=False,
        mtp_num_hidden_layers=0,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.block_attn_res_block_size = block_attn_res_block_size
        self.router_score_function = router_score_function
        self.router_bias_correction = router_bias_correction
        self.norm_topk_prob = norm_topk_prob
        self.kda_allow_negative_eigenvalues = kda_allow_negative_eigenvalues
        self.max_position_embeddings = max_position_embeddings
        # Normalize legacy checkpoint fields into the v5 serialization schema.
        self.rope_parameters = dict(
            rope_parameters if rope_parameters is not None else rope_scaling or {}
        )
        self.rope_parameters.setdefault(
            "rope_type", self.rope_parameters.get("type", "default")
        )
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)
        self.rope_theta = self.rope_parameters["rope_theta"]
        self.partial_rotary_factor = self.rope_parameters["partial_rotary_factor"]
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        self.output_router_logits = output_router_logits
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.number_of_conv_states = 3
        self.layer_types = (
            list(layer_types)
            if layer_types is not None
            else [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(num_hidden_layers)
            ]
        )
        self._validate_fields()

    def _validate_fields(self):
        block_size = self.block_attn_res_block_size
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size <= 0
        ):
            raise ValueError("block_attn_res_block_size must be a positive integer")
        if (
            self.num_hidden_layers <= 0
            or len(self.layer_types) != self.num_hidden_layers
        ):
            raise ValueError("layer_types must contain one entry per hidden layer")
        if set(self.layer_types) - {"linear_attention", "full_attention"}:
            raise ValueError(
                "AliceAI only supports linear_attention and full_attention"
            )
        if self.linear_conv_kernel_dim < 2:
            raise ValueError("linear_conv_kernel_dim must be at least 2")
        if (
            self.linear_num_key_heads != self.linear_num_value_heads
            or self.linear_key_head_dim != self.linear_value_head_dim
        ):
            raise ValueError(
                "AliceAI KDA requires matching key/value heads and dimensions"
            )
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.router_score_function != "sigmoid" or not self.router_bias_correction:
            raise ValueError(
                "AliceAI requires sigmoid routing with expert correction bias"
            )
        if self.kda_allow_negative_eigenvalues:
            raise ValueError("AliceAI KDA with negative eigenvalues is not supported")
        if self.hidden_act != "silu":
            raise ValueError("AliceAI requires silu activation")

    @property
    def linear_layer_ids(self):
        return [
            i for i, kind in enumerate(self.layer_types) if kind == "linear_attention"
        ]

    @property
    def full_attention_layer_ids(self):
        return [
            i for i, kind in enumerate(self.layer_types) if kind == "full_attention"
        ]

    @property
    def linear_attn_config(self):
        # KimiDeltaAttention consumes this view of AliceAI's flat HF config.
        return {
            "num_heads": self.linear_num_key_heads,
            "head_dim": self.linear_key_head_dim,
            "short_conv_kernel_size": self.linear_conv_kernel_dim,
        }

    @property
    def mamba2_cache_params(self):
        shape = KimiLinearStateShape.create(
            tp_world_size=get_parallel().attn_tp_size,
            num_heads=self.linear_num_key_heads,
            head_dim=self.linear_key_head_dim,
            conv_kernel_size=self.linear_conv_kernel_dim,
        )
        return KimiLinearCacheParams(
            shape=shape,
            layers=self.linear_layer_ids,
            dtype=mamba2_state_dtype(self),
        )


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=AliceAIConfig,
        backend_class_name="sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend",
        arch_names=["AliceAIForCausalLM"],
        support_mamba_cache_extra_buffer=True,
        config_predicate=lambda config: bool(config.linear_layer_ids),
    )
)
