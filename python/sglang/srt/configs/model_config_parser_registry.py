"""Named registry for model-config parsers.

Mirrors the ``LoadFormat.PRIVATE`` escape hatch in
:mod:`sglang.srt.configs.load_config` but registry-shaped, so multiple
plugins can coexist without colliding on a single private import path.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


class ModelConfigParserBase(ABC):
    @abstractmethod
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: Optional[str] = None,
        **kwargs,
    ) -> PretrainedConfig:
        raise NotImplementedError


_MODEL_CONFIG_PARSER_REGISTRY: dict[str, type[ModelConfigParserBase]] = {}


def register_model_config_parser(name: str):
    """Returned instances are freshly constructed on each call -- parsers
    should be stateless or carry only per-instance state."""

    def _wrapper(cls):
        if not issubclass(cls, ModelConfigParserBase):
            raise ValueError("Model-config parser must subclass ModelConfigParserBase.")
        if name in _MODEL_CONFIG_PARSER_REGISTRY:
            logger.warning(
                "Model-config parser %r already registered; overwriting with %s",
                name,
                cls,
            )
        _MODEL_CONFIG_PARSER_REGISTRY[name] = cls
        logger.debug("Registered model-config parser %r -> %s", name, cls.__name__)
        return cls

    return _wrapper


def get_model_config_parser(name: str) -> ModelConfigParserBase:
    """``"auto"`` is not handled here -- the caller must resolve it first."""
    if name not in _MODEL_CONFIG_PARSER_REGISTRY:
        raise ValueError(
            f"Unknown model-config parser {name!r}. "
            f"Registered: {sorted(_MODEL_CONFIG_PARSER_REGISTRY)}"
        )
    return _MODEL_CONFIG_PARSER_REGISTRY[name]()
