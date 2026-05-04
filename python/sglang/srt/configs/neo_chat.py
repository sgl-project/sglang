# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import os
from typing import Union

from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from sglang.utils import logger


class NEOLLMConfig(Qwen3Config):
    """Qwen3 text config with U1's extra 2-D RoPE fields."""

    def __init__(
        self,
        rope_theta_hw: float = 10000.0,
        max_position_embeddings_hw: int = 10000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw


class NEOVisionConfig(PretrainedConfig):
    model_type = "neo_vision"

    def __init__(
        self,
        num_channels: int = 3,
        patch_size: int = 16,
        hidden_size: int = 1024,
        llm_hidden_size: int = 2048,
        downsample_ratio: float = 0.5,
        rope_theta_vision: float = 10000.0,
        max_position_embeddings_vision: int = 10000,
        min_pixels: int = 65536,
        max_pixels: int = 4194304,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.downsample_ratio = downsample_ratio
        self.rope_theta_vision = rope_theta_vision
        self.max_position_embeddings_vision = max_position_embeddings_vision
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> PretrainedConfig:
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path,
            **kwargs,
        )
        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]
        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                "You are using a model of type %s to instantiate a model of "
                "type %s. This can yield errors.",
                config_dict["model_type"],
                cls.model_type,
            )
        return cls.from_dict(config_dict, **kwargs)


class NEOChatConfig(PretrainedConfig):
    model_type = "neo_chat"
    is_composition = True
    sub_configs = {
        "vision_config": NEOVisionConfig,
        "llm_config": NEOLLMConfig,
    }

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora: int = 0,
        use_llm_lora: int = 0,
        downsample_ratio: float = 0.5,
        template: str | None = None,
        img_context_token_id: int = 151669,
        img_start_token_id: int = 151670,
        img_end_token_id: int = 151671,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {"architectures": ["NEOVisionModel"]}
            logger.info(
                "vision_config is None. Initializing NEOVisionConfig with defaults."
            )
        if llm_config is None:
            llm_config = {"architectures": ["Qwen3ForCausalLM"]}
            logger.info("llm_config is None. Initializing NEOLLMConfig with defaults.")

        if isinstance(vision_config, dict):
            self.vision_config = NEOVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if isinstance(llm_config, dict):
            self.llm_config = NEOLLMConfig(**llm_config)
        else:
            self.llm_config = llm_config

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.img_context_token_id = img_context_token_id
        self.img_start_token_id = img_start_token_id
        self.img_end_token_id = img_end_token_id

        self.hidden_size = self.llm_config.hidden_size
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        del decoder
        return self.__dict__.get("llm_config", self)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        output["img_context_token_id"] = self.img_context_token_id
        output["img_start_token_id"] = self.img_start_token_id
        output["img_end_token_id"] = self.img_end_token_id
        return output


AutoConfig.register("neo_chat", NEOChatConfig)
AutoConfig.register("neo_vision", NEOVisionConfig)
