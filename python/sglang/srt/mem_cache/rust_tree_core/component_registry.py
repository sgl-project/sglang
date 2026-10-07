"""Factories that construct native tree components before tree operations begin."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Callable

from sglang.srt.mem_cache.unified_cache.component_factory import (
    resolve_component_factory_keys,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.components.registry import (
    get_python_tree_component,
)
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


@dataclass(frozen=True)
class TreeComponentArgument:
    """Arguments supplied to native component factories."""

    component_type: ComponentType
    params: CacheInitParams
    native_init_params: object
    native_bindings: ModuleType
    is_bigram: bool


TreeComponentFactory = Callable[[TreeComponentArgument], object]
_TREE_COMPONENT_REGISTRY: dict[str, TreeComponentFactory] = {}
_PAIRED_PYTHON_COMPONENT_FACTORIES: dict[str, object] = {}


def register_tree_component(
    name: str, factory: TreeComponentFactory, replace: bool = False
) -> None:
    """Register a native factory paired with the current same-name Python factory."""
    if not name.strip():
        raise ValueError("Rust component name must be non-empty")
    if not callable(factory):
        raise TypeError("Rust component factories must be callable")
    existing = _TREE_COMPONENT_REGISTRY.get(name)
    if existing is not None and existing is not factory and not replace:
        raise ValueError(f"Rust component {name!r} is already registered")
    _TREE_COMPONENT_REGISTRY[name] = factory
    _PAIRED_PYTHON_COMPONENT_FACTORIES[name] = get_python_tree_component(name)


def get_tree_component(name: str) -> TreeComponentFactory | None:
    return _TREE_COMPONENT_REGISTRY.get(name)


def registered_tree_components() -> dict[str, TreeComponentFactory]:
    return dict(_TREE_COMPONENT_REGISTRY)


def supports_tree_component(name: str) -> bool:
    """Check that the native factory is paired with the selected Python cache hooks."""
    python_factory = get_python_tree_component(name)
    return (
        name in _TREE_COMPONENT_REGISTRY
        and python_factory is not None
        and _PAIRED_PYTHON_COMPONENT_FACTORIES.get(name) is python_factory
    )


def resolve_component_factories(
    params: CacheInitParams,
) -> dict[ComponentType, TreeComponentFactory]:
    """Snapshot native factories without loading the extension."""
    if any(
        isinstance(selector, type)
        for selector in (params.component_registry_override or {}).values()
    ):
        raise ValueError(
            "Rust TreeCore does not support class-valued component_registry_override"
        )
    result = {}
    for component_type, key in resolve_component_factory_keys(params).items():
        factory = get_tree_component(key) if isinstance(key, str) else None
        if factory is None:
            raise ValueError(
                f"Rust TreeCore does not support component_registry_override "
                f"{key!r} for {component_type.name}"
            )
        result[component_type] = factory
    return result


def create_tree_component(
    factory: TreeComponentFactory, args: TreeComponentArgument
) -> object:
    """Construct and validate a native component in the selected extension module."""
    component = factory(args)
    if not isinstance(component, args.native_bindings.TreeComponentBinding):
        raise TypeError(
            "Rust component factories must return a native TreeComponentBinding"
        )
    if component.component_type != int(args.component_type):
        raise ValueError(
            f"Rust component factory returned kind {component.component_type}, "
            f"expected {args.component_type.name}"
        )
    if component.is_bigram != args.is_bigram:
        raise ValueError("Rust component factory returned an incompatible key mode")
    return component


def _full_component(args: TreeComponentArgument) -> object:
    return args.native_bindings.TreeComponentBinding.full(
        args.native_init_params, args.is_bigram
    )


def _swa_component(args: TreeComponentArgument) -> object:
    return args.native_bindings.TreeComponentBinding.swa(
        args.native_init_params, args.is_bigram
    )


def _mamba_component(args: TreeComponentArgument) -> object:
    return args.native_bindings.TreeComponentBinding.mamba(
        args.native_init_params, args.is_bigram
    )


_TREE_COMPONENT_REGISTRY.update(
    full=_full_component, swa=_swa_component, mamba=_mamba_component
)
# Built-in native drivers require the original Python cache hooks.
_PAIRED_PYTHON_COMPONENT_FACTORIES.update(
    full=FullComponent, swa=SWAComponent, mamba=MambaComponent
)
