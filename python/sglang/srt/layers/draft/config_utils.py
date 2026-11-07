"""Reading and validating fields of a draft HF config (object or dict); no torch."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Optional


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def text_config(config: Any) -> Any:
    """A multimodal wrapper nests the draft's own fields under text_config."""
    nested = cfg_get(config, "text_config", None)
    return config if nested is None else nested


def nested_dict(config: Any, key: str) -> dict:
    value = cfg_get(config, key, None)
    if value is None:
        return {}
    try:
        return dict(value)
    except Exception:
        return {}


def is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def optional_positive_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if not is_int(value) or value <= 0:
        raise ValueError(
            f"Draft config: {field_name} must be a positive integer, got {value!r}."
        )
    return int(value)


def required_positive_int(value: Any, field_name: str, hint: str) -> int:
    parsed = optional_positive_int(value, field_name)
    if parsed is None:
        raise ValueError(f"Draft config: {field_name} is required {hint}.")
    return parsed


def required_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Draft config: {field_name} must be a bool, got {value!r}.")
    return value
