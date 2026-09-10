# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for the original SenseNova-U1 config module."""

from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    DEFAULT_CFG_INTERVAL,
    DEFAULT_CFG_NORM,
    DEFAULT_ENABLE_TIMESTEP_SHIFT,
    DEFAULT_T_EPS,
    DEFAULT_THINK_MODE,
    DEFAULT_TIMESTEP_SHIFT,
    SENSENOVA_U1_CFG_NORM_CHOICES,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
    SENSENOVA_U1_RESOLUTION_ALIGNMENT,
)

_REGISTRY_EXPORTS = (
    "SENSENOVA_U1_MODEL_IDS",
    "SENSENOVA_U1_ADAPTER_ONLY_MODEL_IDS",
    "is_sensenova_u1_model",
    "is_sensenova_u1_adapter_only_model",
)


def __getattr__(name: str):
    # Reading sampling defaults should not import the pipeline registry.
    if name not in _REGISTRY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from sglang.multimodal_gen import registry

    value = getattr(registry, name)
    globals()[name] = value
    return value


__all__ = [
    "DEFAULT_CFG_INTERVAL",
    "DEFAULT_CFG_NORM",
    "DEFAULT_ENABLE_TIMESTEP_SHIFT",
    "DEFAULT_T_EPS",
    "DEFAULT_THINK_MODE",
    "DEFAULT_TIMESTEP_SHIFT",
    "SENSENOVA_U1_CFG_NORM_CHOICES",
    "SENSENOVA_U1_REQUEST_EXTRA_KEY",
    "SENSENOVA_U1_RESOLUTION_ALIGNMENT",
    *_REGISTRY_EXPORTS,
]
