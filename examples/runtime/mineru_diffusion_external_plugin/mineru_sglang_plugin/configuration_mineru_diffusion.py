"""Config classes for MinerU-Diffusion external adapter."""

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig


class SDARConfig(PretrainedConfig):
    model_type = "sdar"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class MinerUDiffusionConfig(PretrainedConfig):
    model_type = "mineru_diffusion"
    sub_configs = {"vision_config": Qwen2VLVisionConfig, "text_config": SDARConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    architectures = ["MinerUDiffusionForConditionalGeneration"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        language_model_config=None,
        vision_model_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        mask_token_id=151669,
        image_size=512,
        patch_size=16,
        downsample_ratio=0.5,
        vision_projector_type="patch_merger2x",
        vision_select_layer=-2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        kwargs.pop("rm_vit_merger", None)
        top_level_torch_dtype = kwargs.pop("torch_dtype", None)
        if text_config is None:
            text_config = language_model_config
        if vision_config is None:
            vision_config = vision_model_config

        if isinstance(text_config, dict):
            self.text_config = SDARConfig(**text_config)
        elif text_config is None:
            self.text_config = SDARConfig()
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            vision_model_type = vision_config.get("model_type", "")
            if vision_model_type != "qwen2_vl":
                raise ValueError(f"Unsupported vision config type: {vision_model_type}")
            self.vision_config = Qwen2VLVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen2VLVisionConfig()
        else:
            self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.mask_token_id = mask_token_id
        self.image_size = image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.vision_projector_type = vision_projector_type
        self.vision_select_layer = vision_select_layer
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=getattr(self.text_config, "bos_token_id", None),
            eos_token_id=getattr(self.text_config, "eos_token_id", None),
            pad_token_id=getattr(self.text_config, "pad_token_id", None),
            **kwargs,
        )
        self.torch_dtype = getattr(self.text_config, "torch_dtype", top_level_torch_dtype)

        for attr in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "hidden_act",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
            "rope_scaling",
            "attention_bias",
            "attention_dropout",
            "vocab_size",
            "use_sliding_window",
            "sliding_window",
            "max_window_layers",
        ):
            if not hasattr(self, attr):
                setattr(self, attr, getattr(self.text_config, attr, None))

    @property
    def language_model_config(self):
        return self.text_config

    @property
    def vision_model_config(self):
        return self.vision_config

    @property
    def vision_model_type(self):
        return getattr(self.vision_config, "model_type", None)


__all__ = ["MinerUDiffusionConfig", "SDARConfig"]
