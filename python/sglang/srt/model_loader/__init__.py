# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/__init__.py

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from sglang.srt.model_loader.loader import BaseModelLoader, get_model_loader
from sglang.srt.model_loader.utils import (
    get_architecture_class_name,
    get_model_architecture,
)

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig


def get_model(
    *,
    model_config: ModelConfig,
    load_config: LoadConfig,
    device_config: DeviceConfig,
) -> nn.Module:
    loader = get_model_loader(load_config, model_config)
    return loader.load_model(
        model_config=model_config,
        device_config=device_config,
    )


__all__ = [
    "get_model",
    "get_model_loader",
    "BaseModelLoader",
    "get_architecture_class_name",
    "get_model_architecture",
]
