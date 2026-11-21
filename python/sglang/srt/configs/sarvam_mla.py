from transformers.configuration_utils import PretrainedConfig

_DEFAULT_ROPE_SCALING = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "rope_type": "deepseek_yarn",
}


class SarvamMLAConfig(PretrainedConfig):
    """Native copy of the `sarvamai/sarvam-105b*` remote config.

    The checkpoint's `configuration_sarvam_moe.py` overrides
    `convert_rope_params_to_dict` and calls `validate_rope(ignore_keys=...)`,
    which transformers 5.19 no longer accepts. The base class already converts
    `rope_scaling` into `rope_parameters`, so no override is needed here.
    """

    model_type = "sarvam_mla"

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        intermediate_size: int = 16384,
        moe_intermediate_size: int = 2048,
        num_experts: int = 128,
        num_experts_per_tok: int = 8,
        num_shared_experts: int = 1,
        first_k_dense_replace: int = 1,
        num_attention_heads: int = 64,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 128,
        kv_lora_rank: int = 512,
        v_head_dim: int = 128,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling: dict = None,
        attention_dropout: float = 0.0,
        output_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        use_cache: bool = True,
        use_qk_norm: bool = True,
        moe_router_enable_expert_bias: bool = True,
        routed_scaling_factor: float = 2.5,
        output_router_logits: bool = False,
        tie_word_embeddings: bool = False,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        embedding_dropout: float = 0.0,
        initializer_range: float = 0.006,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings

        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_rope_head_dim + qk_nope_head_dim
        self.head_dim = kv_lora_rank + qk_rope_head_dim

        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.first_k_dense_replace = first_k_dense_replace

        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.output_router_logits = output_router_logits

        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.embedding_dropout = embedding_dropout
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act
        self.use_cache = use_cache
        self.use_qk_norm = use_qk_norm

        self.rope_theta = rope_theta
        self.default_theta = 10000.0
        self.rope_scaling = (
            dict(_DEFAULT_ROPE_SCALING) if rope_scaling is None else rope_scaling
        )

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
