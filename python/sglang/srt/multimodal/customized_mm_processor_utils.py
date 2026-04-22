from typing import Dict, Type

from transformers import PretrainedConfig, ProcessorMixin

# Useful for registering a custom processor different from Hugging Face's default.
_CUSTOMIZED_MM_PROCESSOR: Dict[str, Type[ProcessorMixin]] = dict()


def register_customized_processor(
    processor_class: Type[ProcessorMixin],
):
    """Class decorator that maps a config class's model_type field to a customized processor class.

    Args:
        processor_class: A processor class that inherits from ProcessorMixin

    Example:
        ```python
        @register_customized_processor(MyCustomProcessor)
        class MyModelConfig(PretrainedConfig):
            model_type = "my_model"

        ```
    """

    def decorator(config_class: PretrainedConfig):
        if not hasattr(config_class, "model_type"):
            raise ValueError(
                f"Class {config_class.__name__} with register_customized_processor should "
                f"have a 'model_type' class attribute."
            )
        _CUSTOMIZED_MM_PROCESSOR[config_class.model_type] = processor_class
        return config_class

    return decorator
