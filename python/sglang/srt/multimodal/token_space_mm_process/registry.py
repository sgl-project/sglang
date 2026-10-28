import importlib
import inspect
import pkgutil

from sglang.srt.multimodal.token_space_mm_process import models
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)
from sglang.srt.multimodal.token_space_mm_process.serving_spec import (
    TokenSpaceServingSpec,
)

# hf_config.architectures name -> token-space strategy class.
TOKEN_SPACE_STRATEGY_MAPPING = {}
# hf_config.architectures name -> token-space serving spec class.
TOKEN_SPACE_SERVING_SPEC_MAPPING = {}


def register_token_space_classes(module) -> None:
    """Register the strategies and serving specs a module defines by their architectures."""
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module.__name__:
            continue
        if issubclass(cls, TokenSpaceProcessStrategy):
            mapping = TOKEN_SPACE_STRATEGY_MAPPING
        elif issubclass(cls, TokenSpaceServingSpec):
            mapping = TOKEN_SPACE_SERVING_SPEC_MAPPING
        else:
            continue
        for arch in cls.models:
            mapping[arch] = cls


def strategy_for(hf_config):
    """The registered strategy that serves `hf_config`, or None."""
    for arch in hf_config.architectures or ():
        if arch in TOKEN_SPACE_STRATEGY_MAPPING:
            return TOKEN_SPACE_STRATEGY_MAPPING[arch]
    return None


def serving_spec_for(hf_config):
    """The registered serving spec for `hf_config`, or the defaults."""
    for arch in hf_config.architectures or ():
        if arch in TOKEN_SPACE_SERVING_SPEC_MAPPING:
            return TOKEN_SPACE_SERVING_SPEC_MAPPING[arch]
    return TokenSpaceServingSpec


for _, _name, _ in pkgutil.iter_modules(models.__path__, models.__name__ + "."):
    register_token_space_classes(importlib.import_module(_name))
