"""Hardware- and shape-keyed launch configs for the fused Triton ops.

Mirrors the FlashInfer autotune-config pattern (a JSON per GPU, keyed on shape):
``autotune_configs/fused_triton_ops/<hardware_key>.json`` holds, per op, an ordered
list of rules ``{"when": {...}, <launch_param>: <val>, ...}`` (first match wins).
Each op passes a portable default to :func:`resolve`, used whenever no rule matches.

``<hardware_key>`` is the normalized device name on CUDA (``NVIDIA_GB300``).
``when`` keys
are ``<param>`` (equality) or ``<param>_{lt,le,ge,gt,eq}``.
"""

from __future__ import annotations

import json
import operator
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

# Shares the curated-config tree with the FlashInfer autotune configs in srt/.
_CONFIG_DIR = (
    Path(__file__).resolve().parents[3]
    / "srt"
    / "autotune_configs"
    / "fused_triton_ops"
)
_METADATA_KEY = "_metadata"

_COMPARATORS = {
    "lt": operator.lt,
    "le": operator.le,
    "ge": operator.ge,
    "gt": operator.gt,
    "eq": operator.eq,
}


@lru_cache(maxsize=None)
def _hardware_key(device_index: int = 0) -> str | None:
    if not torch.cuda.is_available():
        return None
    return "_".join(torch.cuda.get_device_name(device_index).split())


def _make_predicate(when: dict) -> Callable[[dict], bool]:
    checks = []
    for field, expected in when.items():
        param, _, suffix = field.rpartition("_")
        if param and suffix in _COMPARATORS:
            checks.append((param, _COMPARATORS[suffix], expected))
        else:
            checks.append((field, operator.eq, expected))

    def predicate(shape: dict) -> bool:
        return all(p in shape and cmp(shape[p], exp) for p, cmp, exp in checks)

    return predicate


@lru_cache(maxsize=None)
def _load(hardware_key: str | None) -> dict[str, list[tuple[Callable, dict]]]:
    if hardware_key is None:
        return {}
    path = _CONFIG_DIR / f"{hardware_key}.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    compiled: dict[str, list[tuple[Callable, dict]]] = {}
    for op_name, rules in data.items():
        if op_name == _METADATA_KEY:
            continue
        compiled[op_name] = [
            (
                _make_predicate(rule.get("when", {})),
                {k: v for k, v in rule.items() if k != "when"},
            )
            for rule in rules
        ]
    return compiled


def _resolve(
    hardware_key: str | None, op_name: str, defaults: dict, shape: dict
) -> dict:
    for predicate, config in _load(hardware_key).get(op_name, ()):
        if predicate(shape):
            return {**defaults, **config}
    return dict(defaults)


def resolve(op_name: str, defaults: dict, *, device_index: int = 0, **shape) -> dict:
    """First matching rule for this GPU + ``shape``, merged over ``defaults``."""
    return _resolve(_hardware_key(device_index), op_name, defaults, shape)


def split_extra(config: dict, consumed: tuple[str, ...]) -> dict:
    """Config params not consumed positionally => extra Triton kwargs (drop None)."""
    return {k: v for k, v in config.items() if k not in consumed and v is not None}
