import importlib
import inspect
import logging
import pkgutil

from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)
from sglang.srt.multimodal.token_space_mm_process.serving_spec import (
    TokenSpaceServingSpec,
)

logger = logging.getLogger(__name__)

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


def register_token_space_models() -> None:
    """Register the strategies and serving specs of every token_space_mm_process.models module.

    Serving and trainers call it before `strategy_for`. A module that fails to import
    is skipped with a warning, so one broken model cannot stop the others.
    """
    from sglang.srt.multimodal.token_space_mm_process import models

    for _, name, _ in pkgutil.iter_modules(models.__path__, models.__name__ + "."):
        try:
            module = importlib.import_module(name)
        except Exception as e:
            logger.warning(f"Ignore import error when loading {name}: {e}")
            continue
        register_token_space_classes(module)
