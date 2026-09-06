# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type

from sglang.srt.layers.quantization.registry import (
    CPU_SUPPORTED_METHOD_SPECS,
    PLATFORM_OVERRIDE_SPECS,
    QUANTIZATION_METHOD_SPECS,
    all_config_class_specs,
)

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

# Importing this package used to import all 28 config modules, so asking
# "which methods exist" -- or resolving any single one of them -- pulled in
# every third-party quantization dependency. Resolution of a spec to its config
# class is deferred to first use instead, which is why the tables below are
# lazy mappings rather than plain dicts. Keep this module's own imports free of
# torch and of any config module.


def _resolve_spec(spec: str) -> Type["QuantizationConfig"]:
    module_path, _, class_name = spec.rpartition(":")
    return getattr(importlib.import_module(module_path), class_name)


def _platform_conditions() -> Dict[str, Callable[[], bool]]:
    from sglang.srt.utils import (
        is_cpu,
        is_cuda,
        is_gfx95_supported,
        is_mps,
        is_npu,
        is_xpu,
    )

    return {
        "mxfp4_capable": lambda: (
            is_cpu() or is_cuda() or is_gfx95_supported() or is_xpu()
        ),
        "npu": is_npu,
        "xpu": is_xpu,
        "mps": is_mps,
    }


def _active_method_specs() -> Dict[str, str]:
    """Base specs with this platform's overrides applied, in table order."""
    conditions = _platform_conditions()
    unknown = [c for c, _ in PLATFORM_OVERRIDE_SPECS if c not in conditions]
    if unknown:
        raise KeyError(
            f"PLATFORM_OVERRIDE_SPECS names condition(s) {unknown} that "
            f"_platform_conditions() does not define; add them there. "
            f"Known conditions: {sorted(conditions)}"
        )
    specs = dict(QUANTIZATION_METHOD_SPECS)
    for condition, overrides in PLATFORM_OVERRIDE_SPECS:
        if conditions[condition]():
            specs.update(overrides)
    return specs


class _LazyMethodMap(Mapping):
    """Maps method name -> config class, importing a config on first lookup.

    Reads like the plain dict it replaces (`in`, `[...]`, `.keys()`,
    `.items()`, `[*...]`), and preserves its iteration order. Membership tests
    and key iteration resolve nothing; `[...]` resolves the one entry asked
    for, and only `.items()` / `.values()` force every config module to
    import.
    """

    def __init__(self, specs_factory: Callable[[], Dict[str, str]]) -> None:
        self._specs_factory = specs_factory
        self._specs: Optional[Dict[str, str]] = None
        self._resolved: Dict[str, Type["QuantizationConfig"]] = {}

    @property
    def specs(self) -> Dict[str, str]:
        if self._specs is None:
            self._specs = self._specs_factory()
        return self._specs

    def __getitem__(self, name: str) -> Type["QuantizationConfig"]:
        if name not in self._resolved:
            self._resolved[name] = _resolve_spec(self.specs[name])
        return self._resolved[name]

    def __contains__(self, name: object) -> bool:
        # Must not fall through to __getitem__: `Mapping.__contains__` would
        # import a config module just to answer a membership test, and would
        # let an ImportError escape a check expected to be exception-free.
        return name in self.specs

    def __iter__(self) -> Iterator[str]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self.specs)!r})"


# Base quantization methods, plus this platform's overrides.
BASE_QUANTIZATION_METHODS: Mapping[str, Type["QuantizationConfig"]] = _LazyMethodMap(
    _active_method_specs
)

# Subset of the above supported on CPU with AMX.
CPU_QUANTIZATION_METHODS: Mapping[str, Type["QuantizationConfig"]] = _LazyMethodMap(
    lambda: dict(CPU_SUPPORTED_METHOD_SPECS)
)

QUANTIZATION_METHODS = BASE_QUANTIZATION_METHODS

_CONFIG_CLASS_SPECS = all_config_class_specs()


def __getattr__(name: str) -> object:
    """Serve the config classes the package has always re-exported.

    Keeps `from sglang.srt.layers.quantization import Fp8Config` (and every
    other `*Config`) working now that they are no longer imported eagerly.
    Unknown names must raise `AttributeError` so `from ... import <submodule>`
    still falls through to the import machinery.
    """
    if name == "QuantizationConfig":
        from sglang.srt.layers.quantization.base_config import QuantizationConfig

        return QuantizationConfig
    spec = _CONFIG_CLASS_SPECS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _resolve_spec(spec)


def __dir__() -> list[str]:
    return sorted(
        [*globals(), *_CONFIG_CLASS_SPECS, "QuantizationConfig"],
    )


def get_quantization_config(quantization: str) -> Type["QuantizationConfig"]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    from sglang.srt.platforms import current_platform
    from sglang.srt.utils import cpu_has_amx_support, is_cpu

    if is_cpu() and cpu_has_amx_support():
        if quantization not in CPU_QUANTIZATION_METHODS:
            raise ValueError(
                f"Invalid quantization method on CPU: {quantization}. "
                f"Available methods on CPU: {list(CPU_QUANTIZATION_METHODS.keys())}"
            )
        else:
            return CPU_QUANTIZATION_METHODS[quantization]

    if current_platform.is_out_of_tree():
        config = current_platform.get_quantization_config(quantization)

        # If the platform has a quantization config, use it else use the default
        if config is not None:
            return config

    return QUANTIZATION_METHODS[quantization]
