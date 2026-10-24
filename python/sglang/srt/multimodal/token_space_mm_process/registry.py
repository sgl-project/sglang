import importlib
import inspect
import pkgutil

from sglang.srt.multimodal.token_space_mm_process import models
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)

# hf_config.architectures name -> token-space strategy class.
TOKEN_SPACE_STRATEGY_MAPPING = {}


def register_strategies(module) -> None:
    """Register the strategies a module defines by the architectures they serve."""
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(cls, TokenSpaceProcessStrategy)
            and cls.__module__ == module.__name__
        ):
            for arch in cls.models:
                TOKEN_SPACE_STRATEGY_MAPPING[arch] = cls


def strategy_for(hf_config):
    """The registered strategy that serves `hf_config`, or None."""
    for arch in hf_config.architectures or ():
        if arch in TOKEN_SPACE_STRATEGY_MAPPING:
            return TOKEN_SPACE_STRATEGY_MAPPING[arch]
    return None


for _, _name, _ in pkgutil.iter_modules(models.__path__, models.__name__ + "."):
    register_strategies(importlib.import_module(_name))
