# SPDX-License-Identifier: Apache-2.0
"""Configuration for the Cosmos3 Reasoner (understanding tower).

The Cosmos3 unified checkpoint stores a Qwen3-VL understanding tower alongside
a generation (diffusion) tower. The Reasoner only serves the understanding
tower, so it reuses the Qwen3-VL config schema and just declares its own
``model_type`` so ``AutoConfig`` can resolve the checkpoint.
"""

from typing import Optional, Type

from transformers import PretrainedConfig

from sglang.srt.configs.qwen3_vl import Qwen3VLConfig


class Cosmos3Config(Qwen3VLConfig):
    model_type = "cosmos3_omni"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The Qwen3-VL inference stack accesses ``config.vision_config`` and
        # ``config.text_config`` as objects (e.g. ``config.vision_config.hidden_size``,
        # ``config.vision_config.deepstack_visual_indexes``). Some transformers
        # versions leave these sub-configs
        # as raw dicts after construction, which would raise
        # ``'dict' object has no attribute 'hidden_size'`` at model init. Coerce
        # any dict-valued sub-config into its proper config object so the model
        # loads regardless of the installed transformers version.
        for attr, sub_cls in self.sub_configs.items():
            sub = getattr(self, attr, None)
            if isinstance(sub, dict):
                setattr(self, attr, sub_cls(**sub))


def _coerce_sub_config(
    value: Optional[object],
    config_cls: Type[PretrainedConfig],
) -> PretrainedConfig:
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, dict):
        return config_cls(**value)
    if isinstance(value, PretrainedConfig):
        return value
    raise TypeError(f"Unsupported sub-config type: {type(value)!r}")


def _normalize_edge_rope_parameters(value: Optional[dict]) -> Optional[dict]:
    if value is None:
        return None
    rope_parameters = dict(value)
    mrope_section = rope_parameters.get("mrope_section")
    if mrope_section is not None:
        rope_parameters["mrope_section"] = list(mrope_section)
        rope_parameters.setdefault("mrope_interleaved", True)
    return rope_parameters


class Cosmos3EdgeTextConfig(PretrainedConfig):
    model_type = "cosmos3_edge_text"
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 2048,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "relu2",
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        rope_parameters: Optional[dict] = None,
        rope_scaling: Optional[dict] = None,
        use_cache: bool = True,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = 11,
        pad_token_id: Optional[int] = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.use_cache = use_cache

        if rope_parameters is None:
            rope_parameters = rope_scaling
        if rope_parameters is None:
            rope_parameters = {
                "mrope_section": [24, 20, 20],
                "rope_theta": 100000000,
                "rope_type": "default",
            }
        rope_parameters = _normalize_edge_rope_parameters(rope_parameters)
        self.rope_parameters = rope_parameters
        # SGLang's RoPE factory accepts the v5-style rope_parameters schema, but
        # several generic paths still probe rope_scaling.
        self.rope_scaling = _normalize_edge_rope_parameters(
            rope_scaling if rope_scaling is not None else rope_parameters
        )


class Cosmos3EdgeVisionConfig(PretrainedConfig):
    model_type = "cosmos3_edge_vision"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        num_patches: int = 256,
        patch_size: int = 16,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout


class Cosmos3EdgeProjectorConfig(PretrainedConfig):
    model_type = "cosmos3_edge_projector"

    def __init__(
        self,
        input_hidden_size: int = 1152,
        merger_intermediate_size: int = 11520,
        out_hidden_size: int = 2048,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_hidden_size = input_hidden_size
        self.merger_intermediate_size = merger_intermediate_size
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.use_postshuffle_norm = use_postshuffle_norm


class Cosmos3EdgeConfig(PretrainedConfig):
    model_type = "cosmos3_edge"
    sub_configs = {
        "text_config": Cosmos3EdgeTextConfig,
        "vision_config": Cosmos3EdgeVisionConfig,
        "projector_config": Cosmos3EdgeProjectorConfig,
    }

    def __init__(
        self,
        text_config: Optional[object] = None,
        vision_config: Optional[object] = None,
        projector_config: Optional[object] = None,
        image_token_id: int = 19,
        video_token_id: int = 18,
        vision_start_token_id: int = 20,
        vision_end_token_id: int = 21,
        tie_word_embeddings: bool = False,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.text_config = _coerce_sub_config(text_config, Cosmos3EdgeTextConfig)
        self.vision_config = _coerce_sub_config(vision_config, Cosmos3EdgeVisionConfig)
        self.projector_config = _coerce_sub_config(
            projector_config, Cosmos3EdgeProjectorConfig
        )

        # Qwen-style multimodal processing reads these from vision_config, while
        # the Cosmos3-Edge checkpoint stores them under projector_config.
        self.vision_config.spatial_merge_size = self.projector_config.spatial_merge_size
        self.vision_config.temporal_patch_size = 1
        self.vision_config.out_hidden_size = self.projector_config.out_hidden_size

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings

        if getattr(self, "architectures", None) is None:
            self.architectures = ["Cosmos3EdgeForConditionalGeneration"]

        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            parent_value = getattr(self, attr, None)
            text_value = getattr(self.text_config, attr, None)
            if parent_value is None and text_value is not None:
                setattr(self, attr, text_value)
            elif parent_value is not None and text_value is None:
                setattr(self.text_config, attr, parent_value)
        if not hasattr(self.text_config, "tie_word_embeddings"):
            self.text_config.tie_word_embeddings = tie_word_embeddings
