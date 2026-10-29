import importlib
import inspect
import logging
import pkgutil

from sglang.srt.multimodal.token_space_mm_process import models
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)

logger = logging.getLogger(__name__)

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


def register_models_package(package) -> None:
    """Register the strategies of every module in `package`.

    As import_processors does for legacy processors, a module that fails to import
    is skipped with a warning, so one broken model cannot stop every server.
    """
    for _, name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        try:
            module = importlib.import_module(name)
        except Exception as e:
            logger.warning(f"Ignore import error when loading {name}: {e}")
            continue
        register_strategies(module)


register_models_package(models)
