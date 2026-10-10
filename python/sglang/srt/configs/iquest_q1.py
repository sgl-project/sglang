from typing import List, Optional

from transformers import PretrainedConfig


class IQuestQ1Config(PretrainedConfig):
    model_type = "iquest_q1"

    def __init__(
        self,
        vocab_size: int = 160000,
        hidden_size: int = 3072,
        dense_intermediate_size: int = 12288,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 88,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 524288,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: Optional[int] = None,
        eos_token_id: int = 0,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rotary_dim: int = 32,
        rope_parameters: Optional[dict] = None,
        no_rope_layers: Optional[List[int]] = None,
        swa_rope_theta: float = 10000.0,
        sliding_window: int = 4096,
        use_hybrid_layers: bool = True,
        use_sliding_window: bool = True,
        first_layers_types: Optional[List[str]] = None,
        hybrid_layers_types_block: Optional[List[str]] = None,
        num_hybrid_layers_block: int = 21,
        last_layers_types: Optional[List[str]] = None,
        mlp_only_layers: Optional[List[int]] = None,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        moe_router_dtype: str = "fp32",
        enable_sink_attention: bool = True,
        first_layer_attn_out_scale: float = 1.0,
        first_layer_ffn_out_scale: float = 1.0,
        attn_out_scale: float = 1.0,
        ffn_out_scale: float = 0.53881590608,
        softmax_scale: Optional[float] = None,
        logit_scale: float = 1.0,
        enable_lm_head_fp32: bool = True,
        num_mtp_layers: int = 0,
        **kwargs,
    ):
        if first_layers_types is None:
            first_layers_types = ["full_attention"]
        if hybrid_layers_types_block is None:
            hybrid_layers_types_block = [
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
            ]
        if last_layers_types is None:
            last_layers_types = [
                "full_attention",
                "full_attention",
                "full_attention",
            ]
        if mlp_only_layers is None:
            mlp_only_layers = [0]
        layer_types = (
            list(first_layers_types)
            + list(hybrid_layers_types_block) * num_hybrid_layers_block
            + list(last_layers_types)
        )
        if not use_hybrid_layers:
            layer_types = ["full_attention"] * num_hidden_layers
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                "IQuest Q1 layer type pattern has length "
                f"{len(layer_types)}, expected {num_hidden_layers}"
            )
        if any(t not in ("full_attention", "sliding_attention") for t in layer_types):
            raise ValueError(
                "IQuest Q1 supports full_attention and sliding_attention layers"
            )
        if not use_sliding_window:
            layer_types = ["full_attention"] * num_hidden_layers
        if rotary_dim is not None and (
            rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2
        ):
            raise ValueError("rotary_dim must be even, positive, and at most head_dim")
        if moe_router_dtype != "fp32":
            raise ValueError("IQuest Q1 requires moe_router_dtype='fp32'.")
        if kwargs.get("use_over_encoding", False):
            raise ValueError("IQuest Q1 over-encoding embeddings are not supported.")
        if enable_lm_head_fp32 is not True:
            raise ValueError("IQuest Q1 requires enable_lm_head_fp32=True.")
        kwargs.pop("layer_types", None)
        kwargs.setdefault("architectures", ["IQuestQ1ForCausalLM"])
        kwargs.setdefault("dtype", "bfloat16")
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_scaling=rope_scaling,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dense_intermediate_size = dense_intermediate_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rotary_dim = rotary_dim
        self.swa_rope_theta = swa_rope_theta
        self.sliding_window = sliding_window if use_sliding_window else None
        self.use_hybrid_layers = use_hybrid_layers
        self.use_sliding_window = use_sliding_window
        self.no_rope_layers = list(no_rope_layers or [])
        self.first_layers_types = list(first_layers_types)
        self.hybrid_layers_types_block = list(hybrid_layers_types_block)
        self.num_hybrid_layers_block = num_hybrid_layers_block
        self.last_layers_types = list(last_layers_types)
        self.layer_types = layer_types
        self.hybrid_layer_pattern = [
            int(layer_type == "sliding_attention") for layer_type in layer_types
        ]
        self.is_hybrid_swa = any(self.hybrid_layer_pattern)
        self.mlp_only_layers = list(mlp_only_layers)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_router_dtype = moe_router_dtype
        self.enable_sink_attention = enable_sink_attention
        self.first_layer_attn_out_scale = first_layer_attn_out_scale
        self.first_layer_ffn_out_scale = first_layer_ffn_out_scale
        self.attn_out_scale = attn_out_scale
        self.ffn_out_scale = ffn_out_scale
        self.softmax_scale = softmax_scale
        self.logit_scale = logit_scale
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        self.num_mtp_layers = num_mtp_layers
        self.rope_parameters = dict(rope_parameters or rope_scaling or {})
        self.rope_parameters.setdefault(
            "rope_type", self.rope_parameters.get("type", "default")
        )
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        if rotary_dim is not None:
            self.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim


class IQuestQ1MTPConfig(IQuestQ1Config):
    model_type = "iquest_q1_mtp"

    def __init__(
        self,
        target_config: Optional[dict] = None,
        num_draft_slots: int = 7,
        num_target_layers: Optional[int] = None,
        sliding_window: Optional[int] = 512,
        swa_rope_theta: Optional[float] = 10000.0,
        fp32_residual_connection: bool = True,
        **kwargs,
    ):
        if kwargs.get("num_hidden_layers", 1) != 1:
            raise ValueError("MTP supports one physical draft layer.")
        if num_draft_slots < 1:
            raise ValueError("num_draft_slots must be positive.")
        window = sliding_window or None
        if window is not None and (window < 0 or not swa_rope_theta):
            raise ValueError("A positive draft window requires swa_rope_theta.")
        config = dict(target_config or {})
        if num_target_layers is None:
            num_target_layers = config.get("num_hidden_layers", 88)
        if num_target_layers < 1:
            raise ValueError("num_target_layers must be positive.")
        if config.get("num_hidden_layers", num_target_layers) != num_target_layers:
            raise ValueError("num_target_layers must match target_config.")
        for key in ("model_type", "architectures", "auto_map", "_name_or_path"):
            config.pop(key, None)
        config.update(kwargs)
        config.update(
            architectures=["IQuestQ1MTP"],
            num_hidden_layers=1,
            mlp_only_layers=[],
            no_rope_layers=[],
            use_hybrid_layers=True,
            use_sliding_window=window is not None,
            sliding_window=window,
            first_layers_types=[
                "sliding_attention" if window is not None else "full_attention"
            ],
            hybrid_layers_types_block=[],
            num_hybrid_layers_block=0,
            last_layers_types=[],
        )
        if swa_rope_theta is not None:
            config["swa_rope_theta"] = swa_rope_theta
        super().__init__(**config)
        self.target_config = dict(target_config or {})
        self.num_draft_slots = num_draft_slots
        self.num_target_layers = num_target_layers
        self.fp32_residual_connection = fp32_residual_connection
