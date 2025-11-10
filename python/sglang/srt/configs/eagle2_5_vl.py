from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class Eagle2_5_VLVisionConfig(PretrainedConfig):
    """Vision configuration for Eagle2.5 VL model."""

    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_size=224,
        intermediate_size=4304,
        num_attention_heads=16,
        num_hidden_layers=27,
        num_channels=3,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range


class Eagle2_5_VLTextConfig(PretrainedConfig):
    """Text configuration for Eagle2.5 VL model (based on Qwen3)."""

    model_type = "qwen3"

    def __init__(
        self,
        vocab_size=151680,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        rope_config_validation(self)


class Eagle2_5_VLConfig(PretrainedConfig):
    """Main configuration for Eagle2.5 VL model."""

    model_type = "eagle_2_5_vl"
    sub_configs = {
        "vision_config": Eagle2_5_VLVisionConfig,
        "text_config": Eagle2_5_VLTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=151667,
        video_token_id=151670,
        downsample_ratio=0.5,
        dynamic_image_size=True,
        force_image_size=224,
        max_dynamic_tiles=12,
        min_dynamic_tiles=1,
        pad2square=False,
        select_layer=-4,
        template="qwen2-chat",
        torch_dtype="bfloat16",
        use_backbone_lora=0,
        use_llm_lora=0,
        use_pixel_shuffle=True,
        use_thumbnail=True,
        mlp_connector_layers=2,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_index = image_token_index
        self.video_token_id = video_token_id
        self.downsample_ratio = downsample_ratio
        self.dynamic_image_size = dynamic_image_size
        self.force_image_size = force_image_size
        self.max_dynamic_tiles = max_dynamic_tiles
        self.min_dynamic_tiles = min_dynamic_tiles
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.template = template
        self.torch_dtype = torch_dtype
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.use_pixel_shuffle = use_pixel_shuffle
        self.use_thumbnail = use_thumbnail
        self.mlp_connector_layers = mlp_connector_layers

        super().__init__(**kwargs)
