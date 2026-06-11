# SPDX-License-Identifier: Apache-2.0
"""Shared parsing helpers for diffusion acceleration policy modules.

The CLI accepts loose JSON/YAML/key-value configuration. The helpers in this
file keep that translation consistent across attention, transformer compile,
and custom-op compile policies without importing heavy runtime components.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import addict

TORCH_COMPILE_MODE_ENV = "SGLANG_TORCH_COMPILE_MODE"


def normalize_policy(value: Any, default: str = "off") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "auto" if value else "off"
    return str(value).strip().lower().replace("-", "_")


def get_nested(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def first_present(config: Mapping[str, Any], *keys: str | tuple[str, ...]) -> Any:
    for key in keys:
        value = get_nested(config, *key) if isinstance(key, tuple) else config.get(key)
        if value is not None:
            return value
    return None


def env_or_config(
    env_name: str, acceleration_cfg: Mapping[str, Any], *keys: str | tuple[str, ...]
) -> Any:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value
    return first_present(acceleration_cfg, *keys)


def get_server_policy_configs() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    try:
        server_args = get_global_server_args()
    except ValueError:
        return addict.Dict(), addict.Dict()
    return (
        server_args.attention_backend_config or addict.Dict(),
        server_args.acceleration_config or addict.Dict(),
    )
