# TODO: also move pad_input_ids into this module
import importlib
import logging
import pkgutil
from functools import lru_cache

from transformers import IMAGE_PROCESSOR_MAPPING

from sglang.srt.managers.image_processors.base_image_processor import (
    BaseImageProcessor,
    DummyImageProcessor,
)
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


IMAGE_PROCESSOR_MAPPING = {}


def get_image_processor(
    hf_config, server_args: ServerArgs, processor
) -> BaseImageProcessor:
    for model_cls, processor_cls in IMAGE_PROCESSOR_MAPPING.items():
        if model_cls.__name__ in hf_config.architectures:
            return processor_cls(hf_config, server_args, processor)
    raise ValueError(
        f"No image processor found for architecture: {hf_config.architectures}"
    )


def get_dummy_image_processor():
    return DummyImageProcessor()


@lru_cache()
def import_image_processors():
    package_name = "sglang.srt.managers.image_processors"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning(f"Ignore import error when loading {name}: " f"{e}")
                continue
            if hasattr(module, "ImageProcessorMapping"):
                entry = module.ImageProcessorMapping
                if isinstance(entry, dict):
                    for processor_name, cls in entry.items():
                        IMAGE_PROCESSOR_MAPPING[processor_name] = cls


# also register processors
import_image_processors()
