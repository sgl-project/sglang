"""Translate DeepSeek V4.1 HF configs to the runtime's flat config schema."""

from transformers import DeepseekV3Config, PretrainedConfig

_VISION_FIELDS = {
    "num_hidden_layers": "vision_n_layers",
    "hidden_size": "vision_dim",
    "num_attention_heads": "vision_n_heads",
    "intermediate_size": "vision_inter_dim",
    "patch_size": "vision_patch_size",
    "rope_theta": "vision_rope_theta",
    "downsample_ratio": "vision_downsample_ratio",
    "max_image_tokens": "vision_max_n_token",
    "min_pixels": "vision_min_pixels",
    "max_wh_ratio": "vision_max_wh_ratio",
}


def _config_dict(config):
    return config.to_dict() if isinstance(config, PretrainedConfig) else dict(config)


def normalize_deepseek_v41_config(values):
    values = dict(values)
    text = values.pop("text_config", None)
    vision = values.pop("vision_config", None)
    if text is not None:
        text = _config_dict(text)
        text.pop("model_type", None)
        values = {**text, **values}
    if vision is not None:
        vision = _config_dict(vision)
        for source, target in _VISION_FIELDS.items():
            if source in vision:
                values.setdefault(target, vision[source])
    if "model_type" in values:
        values["model_type"] = "deepseek_v41"
    if values.get("architectures") == ["DeepseekV41ForCausalLM"]:
        values["architectures"] = ["DeepseekV4ForCausalLM"]
    return values


class DeepseekV41Config(DeepseekV3Config):
    # V3 accepts the V4.1 compression ratios; the native V4 config rejects 1/2.
    model_type = "deepseek_v41"
    vision_n_layers = 0
    hc_pre_from_prev_sublayer = True
    q_head_norm = False
    kv_source_layer_ids = ()
    index_source_layer_ids = ()
    candidate_source_layer_id = -1
    candidate_topk_blocks = 0
    candidate_block_size = 0
    engram_layer_ids = ()
    engram_num_embeddings = ()
    engram_max_ngram_size = 1
    engram_vocab_size = 0
    engram_n_heads = 0
    engram_head_dim = 0
    engram_pad_token_id = 2
    engram_compressed_vocab_size = 0

    def __init__(self, **kwargs):
        kwargs = normalize_deepseek_v41_config(kwargs)
        kwargs["model_type"] = "deepseek_v41"
        super().__init__(**kwargs)

    def to_dict(self):
        values = super().to_dict()
        values["model_type"] = "deepseek_v41"
        return values


class DeepseekV41TextConfig(DeepseekV41Config):
    model_type = "deepseek_v41_text"
    # Transformers regenerates a dataclass initializer unless it is explicit.
    __init__ = DeepseekV41Config.__init__


class DeepseekV41VisionConfig(PretrainedConfig):
    model_type = "deepseek_v41_vision"


DEEPSEEK_V41_CONFIG_CLASSES = (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)
