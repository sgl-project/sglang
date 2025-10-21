from transformers.configuration_utils import PretrainedConfig

# Qwen3-30B-A3B / checkpoint defaults
CONFIG_DEFAULTS = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "decoder_sparse_step": 1,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 40960,
    "max_window_layers": 48,
    "mlp_only_layers": [],
    "moe_intermediate_size": 768,
    "norm_topk_prob": True,
    "num_attention_heads": 32,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "num_key_value_heads": 4,
    "output_router_logits": False,
    "pad_token_id": 151643,
    "rms_norm_eps": 1e-06,
    "rope_scaling": False,
    "rope_theta": 1000000.0,
    "router_aux_loss_coef": 0.001,
    "sliding_window": False,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": False,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


class RND1Config(PretrainedConfig):
    """
    Configuration class for RND1 models.

    This configuration extends Qwen3MoeConfig with additional parameters
    specific to the RND1 (Radical Numerics Diffusion v1) architecture.

    Args:
        moe_backend: Backend for MoE computation (not used in SGLang, kept for compatibility)
        num_diffusion_steps: Default number of diffusion steps for generation
        mask_token_id: Token ID used for masking (default: 151669 for Qwen)
        **kwargs: Additional arguments passed to PretrainedConfig
    """

    model_type = "rnd1"

    def __init__(
        self,
        moe_backend: str = "sglang",
        num_diffusion_steps: int = 64,
        mask_token_id: int = 151669,
        **kwargs,
    ):
        # Force non-causal and no caching for RND1
        kwargs["use_cache"] = False
        kwargs["is_causal"] = False

        super().__init__(**kwargs)

        # Set defaults after pretrained init to prevent overrides
        self.set_config_defaults()

        # RND1-specific parameters
        self.moe_backend = moe_backend
        self.num_diffusion_steps = num_diffusion_steps
        self.mask_token_id = mask_token_id

        # Ensure bidirectional attention and no caching
        self.is_causal = False
        self.use_cache = False

    def set_config_defaults(self):
        """
        Ensure model defaults are set according to final training checkpoint.

        Qwen3MoeConfig defaults don't match Qwen/Qwen3-30B-A3B settings from which
        RND1 is derived.
        """
        for k, v in CONFIG_DEFAULTS.items():
            if not hasattr(self, k):
                setattr(self, k, v)
