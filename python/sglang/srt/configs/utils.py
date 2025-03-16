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
    # remove_if_exists(IMAGE_PROCESSOR_MAPPING._config_mapping, config.model_type)
    # remove_if_exists(IMAGE_PROCESSOR_MAPPING._model_mapping, config.model_type)
    # remove_if_exists(IMAGE_PROCESSOR_MAPPING._reverse_config_mapping, config.__name__)
    # remove_if_exists(IMAGE_PROCESSOR_MAPPING_NAMES, config.model_type)
    # remove_if_exists(CONFIG_MAPPING_NAMES, config.model_type)
    # print(IMAGE_PROCESSOR_MAPPING.items())
    # CONFIG_MAPPING_NAMES[config.model_type] = config.__name__
    # MODEL_NAMES_MAPPING[config.model_type] = ""
    # CONFIG_MAPPING[config.model_type] = config
    # CONFIG_MAPPING._extra_content[config.model_type] = config
    # print("222222")
    AutoImageProcessor.register(config, None, image_processor, None, exist_ok=True)
    # IMAGE_PROCESSOR_MAPPING._reverse_config_mapping[config.__name__] = config.model_type
    # IMAGE_PROCESSOR_MAPPING._config_mapping[config.model_type] = config.__name__


def register_processor(config: Type[PretrainedConfig], processor: Type[ProcessorMixin]):
    """
    register customized hf processor while removing hf impl
    """
    # remove_if_exists(PROCESSOR_MAPPING._config_mapping, config.model_type)
    # remove_if_exists(PROCESSOR_MAPPING._model_mapping, config.model_type)
    # remove_if_exists(PROCESSOR_MAPPING_NAMES, config.model_type)
    #
    # PROCESSOR_MAPPING._extra_content[config.model_type] = processor
    # # remove_if_exists(CONFIG_MAPPING_NAMES, config.model_type)
    # CONFIG_MAPPING_NAMES[config.model_type] = config.__name__
    # CONFIG_MAPPING[config.model_type] = config
    # CONFIG_MAPPING._extra_content[config.model_type] = config
    # MODEL_NAMES_MAPPING[config.model_type] = ""
    AutoProcessor.register(config, processor, exist_ok=True)
