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
    import inspect

    params_to_pass = {
        # Before https://github.com/huggingface/transformers/pull/43195/changes,
        # 'image_processor_class' parameter is required.
        # Now it is deprecated.
        "image_processor_class": None,
        "slow_image_processor_class": image_processor,
        "fast_image_processor_class": None,
        "exist_ok": True,
    }

    sig = inspect.signature(AutoImageProcessor.register)
    supported_params = sig.parameters.keys()

    valid_kwargs = {k: v for k, v in params_to_pass.items() if k in supported_params}

    AutoImageProcessor.register(config, **valid_kwargs)


def register_processor(config: Type[PretrainedConfig], processor: Type[ProcessorMixin]):
    """
    register customized hf processor while removing hf impl
    """
    AutoProcessor.register(config, processor, exist_ok=True)
