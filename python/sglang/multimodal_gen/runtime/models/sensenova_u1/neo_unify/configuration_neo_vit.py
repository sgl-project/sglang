# Modified for SGLang; see this directory's README.md for upstream source.

import os
from collections.abc import Sequence
from typing import Any, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


def _as_singleton_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) == 1:
            return _as_singleton_tuple(value[0])
        return tuple(value)
    return (value,)


class NEOVisionConfig(PretrainedConfig):
    model_type = "neo_vision"

    def __init__(
        self,
        num_channels=3,
        patch_size=16,
        hidden_size=1024,
        llm_hidden_size=2048,
        downsample_ratio=0.5,
        rope_theta_vision=10000.0,
        max_position_embeddings_vision=10000,
        min_pixels=65536,
        max_pixels=4194304,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.llm_hidden_size = _as_singleton_tuple(llm_hidden_size)
        self.downsample_ratio = _as_singleton_tuple(downsample_ratio)
        self.rope_theta_vision = rope_theta_vision
        self.max_position_embeddings_vision = max_position_embeddings_vision
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
