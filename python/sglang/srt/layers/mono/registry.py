"""Model architecture -> its mono spec, by import path: config-time code reads a spec without importing the model."""

from __future__ import annotations

import importlib
from typing import Optional

from sglang.srt.layers.mono.spec import MonoSpec

_SPECS: dict[str, str] = {}


def spec_path(architecture: str) -> Optional[str]:
    return _SPECS.get(architecture)


def load_spec(architecture: str) -> Optional[MonoSpec]:
    path = spec_path(architecture)
    if path is None:
        return None
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)
