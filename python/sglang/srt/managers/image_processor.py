# TODO: also move pad_input_ids into this module
import importlib
import inspect
import logging
import pkgutil
from functools import lru_cache
from typing import Union

from torch import Tensor
from transformers import IMAGE_PROCESSOR_MAPPING

from sglang.srt.managers.image_processors.base_image_processor import (
    BaseImageProcessor,
    DummyImageProcessor,
)
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


IMAGE_PROCESSOR_MAPPING = {}


def get_image_processor(hf_config, server_args, processor) -> BaseImageProcessor:
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
                logger.warning(f" Ignore import error when loading {name}: " f"{e}")
                continue
            all_members = inspect.getmembers(module, inspect.isclass)
            classes = [
                member
                for name, member in all_members
                if member.__module__ == module.__name__
            ]
            for cls in classes:
                if issubclass(cls, BaseImageProcessor):
                    for arch in getattr(cls, "models"):
                        IMAGE_PROCESSOR_MAPPING[arch] = cls


# also register processors
import_image_processors()
