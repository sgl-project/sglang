from transformers import PretrainedConfig

from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.configs.qwen3_vl import Qwen3VLVisionConfig


class Qwen3_5VisionConfig(Qwen3VLVisionConfig):
    model_type = "qwen3_5"
    base_config_key = "vision_config"


class Qwen3_5TextConfig(Qwen3NextConfig):
    model_type = "qwen3_5_text"
    base_config_key = "text_config"

    def __init__(
        self,
        **kwargs,
    ):
        # HF Qwen3.5 checkpoints may provide RoPE settings under rope_parameters.
        # Normalize it before parent init so downstream code sees the expected values.
        rope_parameters = kwargs.pop("rope_parameters", None)
        if kwargs.get("rope_scaling") is None and rope_parameters is not None:
            kwargs["rope_scaling"] = rope_parameters

        super().__init__(**kwargs)
        if self.rope_scaling is None:
            self.rope_scaling = rope_parameters or {}

        # Keep both names for compatibility with model code paths that read either.
        self.rope_parameters = rope_parameters or self.rope_scaling


class Qwen3_5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3_5Model`]. It is used to instantiate a
    Qwen3.5 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3.5.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3_5TextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen3_5VisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The start token index to encode the image prompt.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The end token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embeddings.

    ```python
    >>> from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5Config

    >>> # Initializing a Qwen3.5 style configuration
    >>> configuration = Qwen3_5Config()

    >>> # Initializing a model from the Qwen3.5 style configuration
    >>> model = Qwen3_5ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_5"
    sub_configs = {
        "vision_config": Qwen3_5VisionConfig,
        "text_config": Qwen3_5TextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config = super().from_dict(config_dict, **kwargs)
        if isinstance(getattr(config, "vision_config", None), dict):
            config.vision_config = cls.sub_configs["vision_config"](
                **config.vision_config
            )
        if isinstance(getattr(config, "text_config", None), dict):
            config.text_config = cls.sub_configs["text_config"](**config.text_config)
        return config

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


class Qwen3_5MoeVisionConfig(Qwen3_5VisionConfig):
    model_type = "qwen3_5_moe"


class Qwen3_5MoeTextConfig(Qwen3_5TextConfig):
    model_type = "qwen3_5_moe_text"
    base_config_key = "text_config"

    def __init__(self, **kwargs):
        # Save MoE kwargs before calling parent, then pass them through
        norm_topk_prob = kwargs.pop("norm_topk_prob", None)
        decoder_sparse_step = kwargs.pop("decoder_sparse_step", None)
        num_experts = kwargs.pop("num_experts", None)
        num_experts_per_tok = kwargs.pop("num_experts_per_tok", None)
        moe_intermediate_size = kwargs.pop("moe_intermediate_size", None)
        mlp_only_layers = kwargs.pop("mlp_only_layers", None)
        router_aux_loss_coef = kwargs.pop("router_aux_loss_coef", None)
        shared_expert_intermediate_size = kwargs.pop(
            "shared_expert_intermediate_size", None
        )
        layer_types = kwargs.pop("layer_types", None)
        mtp_num_hidden_layers = kwargs.pop("mtp_num_hidden_layers", None)
        mtp_use_dedicated_embeddings = kwargs.pop("mtp_use_dedicated_embeddings", None)
        attn_output_gate = kwargs.pop("attn_output_gate", None)
        full_attention_interval = kwargs.pop("full_attention_interval", None)

        super().__init__(**kwargs)

        # Assign MoE attributes after parent init
        if norm_topk_prob is not None:
            self.norm_topk_prob = norm_topk_prob
        if decoder_sparse_step is not None:
            self.decoder_sparse_step = decoder_sparse_step
        if num_experts is not None:
            self.num_experts = num_experts
        if num_experts_per_tok is not None:
            self.num_experts_per_tok = num_experts_per_tok
        if moe_intermediate_size is not None:
            self.moe_intermediate_size = moe_intermediate_size
        if mlp_only_layers is not None:
            self.mlp_only_layers = mlp_only_layers
        if router_aux_loss_coef is not None:
            self.router_aux_loss_coef = router_aux_loss_coef
        if shared_expert_intermediate_size is not None:
            self.shared_expert_intermediate_size = shared_expert_intermediate_size
        if layer_types is not None:
            self.layer_types = layer_types
        if mtp_num_hidden_layers is not None:
            self.mtp_num_hidden_layers = mtp_num_hidden_layers
        if mtp_use_dedicated_embeddings is not None:
            self.mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings
        if attn_output_gate is not None:
            self.attn_output_gate = attn_output_gate
        if full_attention_interval is not None:
            self.full_attention_interval = full_attention_interval


class Qwen3_5MoeConfig(Qwen3_5Config):
    model_type = "qwen3_5_moe"
    sub_configs = {
        "vision_config": Qwen3_5MoeVisionConfig,
        "text_config": Qwen3_5MoeTextConfig,
    }

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config = super().from_dict(config_dict, **kwargs)
        if isinstance(getattr(config, "vision_config", None), dict):
            config.vision_config = cls.sub_configs["vision_config"](
                **config.vision_config
            )
        if isinstance(getattr(config, "text_config", None), dict):
            config.text_config = cls.sub_configs["text_config"](**config.text_config)
        return config
