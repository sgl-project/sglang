# Copyright 2026 The SGLang team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sglang-side ``PretrainedConfig`` classes for MiniCPM-V 4.6.

Mirrors HF ref ``transformers/models/minicpmv4_6/configuration_minicpmv4_6.py``
so we can register the configs ourselves while transformers main has not
yet shipped native ``MiniCPMV4_6Config`` (lands 5.7+).
"""

from typing import Any, Dict, Optional, Union

from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig


class MiniCPMV4_6VisionConfig(PretrainedConfig):
    model_type = "minicpmv4_6_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 980,
        patch_size: int = 14,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        insert_layer_id: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.insert_layer_id = insert_layer_id


def _resolve_text_config_class(model_type: Optional[str]) -> type:
    """``model_type`` -> registered config class. sglang's ``Qwen3_5TextConfig``
    wins over the stock entry when both exist (it carries ``layers_block_type``
    etc. that the model code reads); ``AutoConfig.register`` doesn't replace
    existing entries so we have to short-circuit here. Note that
    ``CONFIG_MAPPING.get`` returns ``None`` even on hit — go through
    ``__getitem__`` to trigger the lazy class import.
    """
    if model_type == Qwen3_5TextConfig.model_type:
        return Qwen3_5TextConfig
    if model_type and model_type in CONFIG_MAPPING:
        return CONFIG_MAPPING[model_type]
    raise KeyError(f"Unknown text_config model_type: {model_type!r}")


def _build_text_config(
    text_config: Union[None, dict, PretrainedConfig],
) -> PretrainedConfig:
    """Coerce ``text_config`` into the right registered backbone class.

    ``AutoConfig.from_pretrained`` resolves the ``"text_config"`` entry of
    ``sub_configs`` and hands us a pre-built ``PretrainedConfig``; manual
    construction in tests / examples passes a dict or ``None``.
    """
    if text_config is None:
        return _resolve_text_config_class(Qwen3_5TextConfig.model_type)()
    if isinstance(text_config, PretrainedConfig):
        cls = _resolve_text_config_class(getattr(text_config, "model_type", None))
        if isinstance(text_config, cls):
            return text_config
        return cls(**text_config.to_dict())
    if isinstance(text_config, dict):
        cfg = dict(text_config)
        cls = _resolve_text_config_class(cfg.pop("model_type", None))
        return cls(**cfg)
    raise TypeError(f"Unsupported text_config type: {type(text_config)}")


class MiniCPMV4_6Config(PretrainedConfig):
    model_type = "minicpmv4_6"
    # No type annotation: transformers 5+ wraps PretrainedConfig subclasses
    # with @dataclass(kw_only=True), and an annotated mutable default would be
    # rejected as a dataclass field. Matches qwen3_5/qwen3_vl/qwen3_omni.
    sub_configs = {
        "vision_config": MiniCPMV4_6VisionConfig,
        "text_config": AutoConfig,
    }

    def __init__(
        self,
        text_config: Optional[Union[Dict[str, Any], PretrainedConfig]] = None,
        vision_config: Optional[Union[Dict[str, Any], PretrainedConfig]] = None,
        insert_layer_id: int = 6,
        image_size: int = 448,
        drop_vision_last_layer: bool = False,
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        tie_word_embeddings: bool = False,
        downsample_mode: str = "16x",
        merge_kernel_size=(2, 2),
        merger_times: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        if isinstance(vision_config, dict):
            vc = dict(vision_config)
            vc.pop("model_type", None)
            self.vision_config = MiniCPMV4_6VisionConfig(**vc)
        elif vision_config is None:
            self.vision_config = MiniCPMV4_6VisionConfig()
        else:
            self.vision_config = vision_config

        # Mirror the ref ``__post_init__``: keep ``insert_layer_id`` in sync on
        # both the top-level and the vision sub-config.
        self.vision_config.insert_layer_id = insert_layer_id
        self.patch_size = self.vision_config.patch_size

        self.text_config = _build_text_config(text_config)

        self.insert_layer_id = insert_layer_id
        self.image_size = image_size
        self.drop_vision_last_layer = drop_vision_last_layer
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.downsample_mode = downsample_mode
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.merger_times = merger_times

    # ``MiniCPMBaseModel.__init__`` reads ``self.config.hidden_size`` (written
    # against flat 2.6/4.0/4.5 configs) and ``LogitsProcessor.__init__`` reads
    # ``config.vocab_size`` — proxy both to ``text_config`` so we don't have to
    # fork the base class / logits processor.
    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size


__all__ = ["MiniCPMV4_6Config", "MiniCPMV4_6VisionConfig"]
