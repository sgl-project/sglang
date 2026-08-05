# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# Registries for efficiency techniques and per-model specs.
#
# A model registers a ModelSpec factory under its transformer class name (or an
# explicit key), and a new technique registers a factory under its name.
# get_model_spec() resolves a live transformer class the same way
# BlockAdapterRegister.is_supported() does.

from __future__ import annotations

from typing import Callable

from sglang.multimodal_gen.runtime.efficiency.spec import ModelSpec
from sglang.multimodal_gen.runtime.efficiency.technique import Technique

_TECHNIQUES: dict[str, Callable[..., Technique]] = {}
_TRANSFORMS: dict[str, Callable[..., object]] = {}
_MODEL_SPECS: dict[str, Callable[[], ModelSpec]] = {}


def register_technique(name: str):
    def deco(factory: Callable[..., Technique]):
        _TECHNIQUES[name] = factory
        return factory

    return deco


def register_transform(name: str):
    def deco(factory: Callable[..., object]):
        _TRANSFORMS[name] = factory
        return factory

    return deco


def build_transform(name: str, **kwargs):
    if name not in _TRANSFORMS:
        raise KeyError(f"unknown transform {name!r}; registered: {sorted(_TRANSFORMS)}")
    return _TRANSFORMS[name](**kwargs)


def registered_transforms() -> list[str]:
    return sorted(_TRANSFORMS)


def register_model_spec(*keys: str):
    """Register a ModelSpec factory under one or more transformer-class keys."""

    def deco(factory: Callable[[], ModelSpec]):
        for k in keys:
            _MODEL_SPECS[k] = factory
        return factory

    return deco


def build_technique(name: str, **kwargs) -> Technique:
    if name not in _TECHNIQUES:
        raise KeyError(
            f"unknown technique {name!r}; registered: {sorted(_TECHNIQUES)}"
        )
    return _TECHNIQUES[name](**kwargs)


def get_model_spec(transformer_or_key) -> ModelSpec | None:
    """Resolve a ModelSpec from a transformer instance or an explicit key.

    Returns None when no spec is registered (the model exposes no seams ->
    techniques requiring capabilities will be refused by compose()).
    """
    if isinstance(transformer_or_key, str):
        key = transformer_or_key
    else:
        key = type(transformer_or_key).__name__
    factory = _MODEL_SPECS.get(key)
    return factory() if factory is not None else None


def is_supported(transformer_or_key) -> bool:
    return get_model_spec(transformer_or_key) is not None


def registered_techniques() -> list[str]:
    return sorted(_TECHNIQUES)


def registered_models() -> list[str]:
    return sorted(_MODEL_SPECS)
