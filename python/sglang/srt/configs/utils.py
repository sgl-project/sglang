from collections import OrderedDict
from typing import Type

from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    BaseImageProcessor,
    PretrainedConfig,
    ProcessorMixin,
)


def remove_if_exists(mapping, key):
    if key in mapping:
        if isinstance(mapping, OrderedDict):
            mapping.pop(key)
            mapping.popitem(key)


def register_image_processor(
    config: Type[PretrainedConfig], image_processor: Type[BaseImageProcessor]
):
    """
    register customized hf image processor while removing hf impl
    """
    AutoImageProcessor.register(config, None, image_processor, None, exist_ok=True)


def register_processor(config: Type[PretrainedConfig], processor: Type[ProcessorMixin]):
    """
    register customized hf processor while removing hf impl
    """
    AutoProcessor.register(config, processor, exist_ok=True)
