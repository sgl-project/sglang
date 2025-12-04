from transformers import PretrainedConfig
from transformers.configuration_utils import layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation

from sglang.utils import logger


class Qwen3OmniMoeAudioEncoderConfig(PretrainedConfig):
    model_type = "qwen3_omni_moe_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = (
            scale_embedding  # scale factor will be sqrt(d_model) if True
        )
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


class Qwen3OmniMoeVisionEncoderConfig(PretrainedConfig):
    model_type = "qwen3_omni_moe_vision_encoder"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes


class Qwen3OmniMoeTextConfig(PretrainedConfig):
    model_type = "qwen3_omni_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3OmniMoeText`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=3584,
        hidden_size=2048,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        sliding_window=None,
        attention_dropout=0,
        decoder_sparse_step=1,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers


class Qwen3OmniMoeThinkerConfig(PretrainedConfig):
    model_type = "qwen3_omni_moe_thinker"
    attribute_map = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {
        "audio_config": Qwen3OmniMoeAudioEncoderConfig,
        "vision_config": Qwen3OmniMoeVisionEncoderConfig,
        "text_config": Qwen3OmniMoeTextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        vision_config=None,
        text_config=None,
        audio_token_id=151646,
        image_token_id=151655,
        video_token_id=151656,
        position_id_per_seconds=25,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_token_id = user_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range

        if isinstance(vision_config, dict):
            vision_config = Qwen3OmniMoeVisionEncoderConfig(**vision_config)
        elif vision_config is None:
            vision_config = Qwen3OmniMoeVisionEncoderConfig()
        self.vision_config = vision_config

        if isinstance(audio_config, dict):
            audio_config = Qwen3OmniMoeAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3OmniMoeAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen3OmniMoeTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3OmniMoeTextConfig()
        self.text_config = text_config
        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id


class Qwen3OmniMoeTalkerCodePredictorConfig(PretrainedConfig):

    model_type = "qwen3_omni_moe_talker_code_predictor"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3OmniMoeTalkerCodePredictor`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=2048,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=5,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        sliding_window=None,
        layer_types=None,
        attention_dropout=0,
        num_code_groups=32,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                (
                    "sliding_attention"
                    if self.sliding_window is not None and i >= self.max_window_layers
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)
        self.num_code_groups = num_code_groups


class Qwen3OmniMoeTalkerTextConfig(PretrainedConfig):

    model_type = "qwen3_omni_moe_talker_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3OmniMoeTalkerText`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=3072,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        sliding_window=None,
        attention_dropout=0,
        decoder_sparse_step=1,
        moe_intermediate_size=384,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers


class Qwen3OmniMoeTalkerConfig(PretrainedConfig):

    sub_configs = {
        "code_predictor_config": Qwen3OmniMoeTalkerCodePredictorConfig,
        "text_config": Qwen3OmniMoeTalkerTextConfig,
    }

    def __init__(
        self,
        code_predictor_config=None,
        text_config=None,
        num_code_groups=32,
        thinker_hidden_size=2048,
        codec_eos_token_id=4198,
        accept_hidden_layer=18,
        codec_nothink_id=4203,
        codec_think_bos_id=4204,
        codec_think_eos_id=4205,
        codec_pad_id=4196,
        codec_bos_id=4197,
        audio_token_id=151646,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        position_id_per_seconds=25,
        audio_start_token_id=151669,
        speaker_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if code_predictor_config is None:
            code_predictor_config = {}
            self.code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig()
            logger.info(
                "code_predictor_config is None. Initializing code_predictor_config model with default values"
            )
        elif isinstance(code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig):
            self.code_predictor_config = code_predictor_config
        else:
            self.code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig(
                **code_predictor_config
            )

        if text_config is None:
            text_config = {}
            self.text_config = Qwen3OmniMoeTalkerTextConfig()
            logger.info(
                "talker text_config is None. Initializing talker text model with default values"
            )
        elif isinstance(text_config, Qwen3OmniMoeTalkerTextConfig):
            self.text_config = text_config
        else:
            self.text_config = Qwen3OmniMoeTalkerTextConfig(**text_config)
        self.num_code_groups = num_code_groups
        self.thinker_hidden_size = thinker_hidden_size
        self.codec_eos_token_id = codec_eos_token_id
        self.accept_hidden_layer = accept_hidden_layer
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.audio_start_token_id = audio_start_token_id
        self.vision_start_token_id = vision_start_token_id
        self.speaker_id = speaker_id


class Qwen3OmniMoeCode2WavConfig(PretrainedConfig):

    def __init__(
        self,
        codebook_size=2048,
        hidden_size=1024,
        max_position_embeddings=8000,
        rope_theta=10000,
        num_attention_heads=16,
        num_key_value_heads=16,
        attention_bias=False,
        sliding_window=72,
        intermediate_size=3072,
        hidden_act="silu",
        layer_scale_initial_scale=0.01,
        rms_norm_eps=1e-5,
        num_hidden_layers=8,
        num_quantizers=16,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        decoder_dim=1536,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.sliding_window = sliding_window
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.layer_scale_initial_scale = layer_scale_initial_scale
        self.rms_norm_eps = rms_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_quantizers = num_quantizers
        self.upsample_rates = upsample_rates
        self.upsampling_ratios = upsampling_ratios
        self.decoder_dim = decoder_dim
        self.attention_dropout = attention_dropout

    @property
    def layer_types(self):
        """
        All layer in code2wav should be sliding attention
        """
        return ["sliding_attention"] * self.num_hidden_layers


class Qwen3OmniMoeConfig(PretrainedConfig):

    model_type = "qwen3_omni_moe"
    sub_configs = {
        "thinker_config": Qwen3OmniMoeThinkerConfig,
        "talker_config": Qwen3OmniMoeTalkerConfig,
        "code2wav_config": Qwen3OmniMoeCode2WavConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        talker_config=None,
        code2wav_config=None,
        enable_audio_output=True,
        im_start_token_id=151644,
        im_end_token_id=151645,
        tts_pad_token_id=151671,
        tts_bos_token_id=151672,
        tts_eos_token_id=151673,
        system_token_id=8948,
        user_token_id=872,
        assistant_token_id=77091,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if thinker_config is None:
            thinker_config = {}
            logger.info(
                "thinker_config is None. Initializing thinker model with default values"
            )

        if talker_config is None:
            talker_config = {}
            logger.info(
                "talker_config is None. Initializing talker model with default values"
            )

        if code2wav_config is None:
            code2wav_config = {}
            logger.info(
                "code2wav_config is None. Initializing code2wav model with default values"
            )

        self.thinker_config = Qwen3OmniMoeThinkerConfig(**thinker_config)
        self.talker_config = Qwen3OmniMoeTalkerConfig(**talker_config)
        self.code2wav_config = Qwen3OmniMoeCode2WavConfig(**code2wav_config)
        self.enable_audio_output = enable_audio_output
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.system_token_id = system_token_id
        self.user_token_id = user_token_id
        self.assistant_token_id = assistant_token_id

    def get_text_config(self, decoder=False) -> "PretrainedConfig":
        """
        Returns the config that is meant to be used with text IO. On most models, it is the original config instance
        itself. On specific composite models, it is under a set of valid names.

        Args:
            decoder (`Optional[bool]`, *optional*, defaults to `False`):
                If set to `True`, then only search for decoder config names.
        """
        # Overridden for deeply nested config like Qwen2-Omni. We don't have any omni model
        # except for Qwen yet. This has to be generalized if more deeply nested configs are
        # added. NOTE: currently method used only by vLLM
        return self.thinker_config.get_text_config()
