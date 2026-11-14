"""Native nested Hugging Face configuration for Muse Spark."""

from transformers import CONFIG_MAPPING, PretrainedConfig


class MuseSparkTextConfig(PretrainedConfig):
    model_type = "muse_spark_text"

    def __init__(
        self,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=200000,
        eos_token_id=200001,
        final_logit_softcapping=20.0,
        head_dim=64,
        hidden_act="silu",
        hidden_size=8192,
        initializer_range=0.02,
        intermediate_size=16384,
        layer_rope_theta=None,
        layer_types=None,
        max_position_embeddings=1048576,
        mlp_layer_types=None,
        moe_hidden_size=4096,
        moe_intermediate_size=4096,
        num_attention_heads=128,
        num_experts_per_tok=8,
        num_hidden_layers=62,
        num_key_value_heads=16,
        num_local_experts=256,
        num_shared_experts=0,
        output_multiplier=0.17677669529663687,
        pad_token_id=200018,
        post_norm_eps=1e-08,
        qk_scale_factor=5.4730064863875,
        residual_gate_temperature=0.3,
        rms_norm_eps=1e-05,
        rope_parameters=None,
        routed_scaling_factor=1.0,
        sliding_window=2048,
        tie_word_embeddings=False,
        use_cache=True,
        vocab_size=202048,
        **kwargs,
    ):
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.final_logit_softcapping = final_logit_softcapping
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_types = (
            layer_types
            if layer_types is not None
            else ["full_attention"] * num_hidden_layers
        )
        self.max_position_embeddings = max_position_embeddings
        self.mlp_layer_types = (
            mlp_layer_types
            if mlp_layer_types is not None
            else ["sparse"] * num_hidden_layers
        )
        self.moe_hidden_size = moe_hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.num_local_experts = num_local_experts
        self.num_shared_experts = num_shared_experts
        self.output_multiplier = output_multiplier
        self.post_norm_eps = post_norm_eps
        self.qk_scale_factor = qk_scale_factor
        self.residual_gate_temperature = residual_gate_temperature
        self.rms_norm_eps = rms_norm_eps
        self.rope_parameters = (
            rope_parameters
            if rope_parameters is not None
            else {"rope_theta": 500000.0, "rope_type": "default"}
        )
        self.routed_scaling_factor = routed_scaling_factor
        self.sliding_window = sliding_window
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.layer_rope_theta = (
            layer_rope_theta
            if layer_rope_theta is not None
            else [self.rope_parameters["rope_theta"]] * num_hidden_layers
        )
        for name in ("layer_types", "layer_rope_theta", "mlp_layer_types"):
            if len(getattr(self, name)) != num_hidden_layers:
                raise ValueError(f"{name} must have num_hidden_layers entries")
        if set(self.layer_types) - {"sliding_attention", "full_attention"}:
            raise ValueError("Unsupported Muse attention layer_types")
        if set(self.mlp_layer_types) - {"sparse", "dense"}:
            raise ValueError("Unsupported Muse mlp_layer_types")
        if self.rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("Muse text attention supports default RoPE only")
        if attention_bias or hidden_act != "silu" or num_shared_experts:
            raise ValueError(
                "Muse supports bias-free attention, silu and routed experts only"
            )
        kwargs.setdefault("dtype", "bfloat16")
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_hybrid_swa(self):
        return "sliding_attention" in self.layer_types


class MuseSparkVisionConfig(PretrainedConfig):
    model_type = "muse_spark_vision"

    def __init__(
        self,
        hidden_act="gelu",
        hidden_size=1536,
        intermediate_size=8960,
        layer_norm_eps=1e-05,
        layer_types=None,
        max_position_embeddings=1024,
        merge_size=2,
        num_attention_heads=16,
        num_hidden_layers=50,
        patch_size=14,
        patch_temporal=2,
        pos_emb_height=32,
        pos_emb_width=32,
        rope_parameters=None,
        **kwargs,
    ):
        # Initialize HF's common fields before adding the vision schedule: its
        # generic layer-type validator only knows text attention types.
        super().__init__(**kwargs)
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.layer_types = (
            layer_types
            if layer_types is not None
            else ["full_attention"] * num_hidden_layers
        )
        self.max_position_embeddings = max_position_embeddings
        self.merge_size = merge_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.patch_size = patch_size
        self.patch_temporal = patch_temporal
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width
        self.rope_parameters = (
            rope_parameters
            if rope_parameters is not None
            else {"rope_theta": 10000, "rope_type": "axial"}
        )
        if len(self.layer_types) != num_hidden_layers or set(self.layer_types) - {
            "window_attention",
            "full_attention",
        }:
            raise ValueError("Invalid Muse vision layer_types")


class MuseSparkAudioConfig(PretrainedConfig):
    model_type = "muse_spark_audio"

    def __init__(
        self,
        conv_kernel_size=7,
        depths=None,
        hidden_act="gelu",
        hidden_sizes=None,
        initializer_range=0.02,
        mlp_ratio=4.0,
        num_hidden_layers=4,
        out_hidden_size=8192,
        patch_sizes=None,
        rms_norm_eps=1e-06,
        **kwargs,
    ):
        self.conv_kernel_size = conv_kernel_size
        self.depths = depths if depths is not None else [2, 2, 2, 2]
        self.hidden_act = hidden_act
        self.hidden_sizes = (
            hidden_sizes if hidden_sizes is not None else [256, 512, 1024, 2048]
        )
        self.initializer_range = initializer_range
        self.mlp_ratio = mlp_ratio
        self.num_hidden_layers = num_hidden_layers
        self.out_hidden_size = out_hidden_size
        self.patch_sizes = patch_sizes if patch_sizes is not None else [16, 6, 5, 4]
        self.rms_norm_eps = rms_norm_eps
        super().__init__(**kwargs)


class MuseSparkConfig(PretrainedConfig):
    model_type = "muse_spark"
    sub_configs = {
        "text_config": MuseSparkTextConfig,
        "vision_config": MuseSparkVisionConfig,
        "audio_config": MuseSparkAudioConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        out_hidden_size=6144,
        projector_hidden_size=4096,
        projector_hidden_act="gelu",
        image_token_id=200092,
        video_token_id=200091,
        audio_token_id=200038,
        **kwargs,
    ):
        kwargs.setdefault("dtype", "bfloat16")
        super().__init__(**kwargs)
        for name, value in (
            ("text_config", text_config),
            ("vision_config", vision_config),
            ("audio_config", audio_config),
        ):
            if not isinstance(value, PretrainedConfig):
                value = self.sub_configs[name](**(value or {}))
            setattr(self, name, value)
        self.text_config.dtype = self.dtype
        self.out_hidden_size = out_hidden_size
        self.projector_hidden_size = projector_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id


def should_skip_weight_loading(name, config):
    if "rotary_emb" in name or "num_tokens_per_expert" in name:
        return True
    if name.endswith(".mlp.norm_weight") or "embed_norm.weight" in name:
        return True
    return config.tie_word_embeddings and name == "lm_head.weight"


for _config_class in (
    MuseSparkTextConfig,
    MuseSparkVisionConfig,
    MuseSparkAudioConfig,
    MuseSparkConfig,
):
    CONFIG_MAPPING.register(_config_class.model_type, _config_class, exist_ok=True)
