from typing import Type

from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    BaseImageProcessor,
    PretrainedConfig,
    ProcessorMixin,
)


def register_image_processor(
    config: Type[PretrainedConfig], image_processor: Type[BaseImageProcessor]
):
    """
    register customized hf image processor while removing hf impl
    """
    try:
        # Attempt to register with named parameter for transformers >= 5.0
        AutoImageProcessor.register(
            config, 
            slow_image_processor_class=image_processor, 
            exist_ok=True
        )
    except TypeError:
        # Fallback: use only positional args without exist_ok conflict
        try:
            AutoImageProcessor.register(config, image_processor, None, exist_ok=True)
        except TypeError:
            # Last resort: register without exist_ok
            AutoImageProcessor.register(config, image_processor, None)


def register_processor(config: Type[PretrainedConfig], processor: Type[ProcessorMixin]):
    """
    register customized hf processor while removing hf impl
    """
    try:
        AutoProcessor.register(config, processor, exist_ok=True)
    except TypeError:
        # Fallback: try alternative signature
        try:
            AutoProcessor.register(config, None, processor, exist_ok=True)
        except TypeError:
            # Last resort: register without exist_ok
            AutoProcessor.register(config, None, processor)
