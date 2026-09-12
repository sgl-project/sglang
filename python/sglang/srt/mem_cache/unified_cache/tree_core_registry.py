"""Registry for pluggable TreeCore implementations.

The unified cache constructs its TreeCore through `create_tree_core`, selected
by SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND (default "python"). To plug in a custom
implementation, register it under a string name via
`register_tree_core_backend(name, factory)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.components import TreeComponent
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
        UnifiedTreeCoreInterface,
    )

TreeCoreFactory = Callable[
    ["CacheInitParams", "dict[ComponentType, TreeComponent]"],
    "UnifiedTreeCoreInterface",
]

_TREE_CORE_REGISTRY: dict[str, TreeCoreFactory] = {}


def register_tree_core_backend(name: str, factory: TreeCoreFactory) -> None:
    """Register a TreeCore factory under `name`."""
    if not name.strip():
        raise ValueError(
            f"register_tree_core_backend: name must be non-empty, got {name!r}"
        )
    if name in _TREE_CORE_REGISTRY:
        raise ValueError(f"register_tree_core_backend: {name!r} is already registered")
    _TREE_CORE_REGISTRY[name] = factory


def get_tree_core_factory(name: str) -> Optional[TreeCoreFactory]:
    return _TREE_CORE_REGISTRY.get(name)


def registered_tree_core_backends() -> list[str]:
    return list(_TREE_CORE_REGISTRY.keys())


def _python_tree_core_factory(
    params: CacheInitParams, components: dict[ComponentType, TreeComponent]
) -> UnifiedTreeCoreInterface:
    """The pure-Python TreeCore."""
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

    return UnifiedTreeCore(params, components)


def _rust_tree_core_factory(
    params: CacheInitParams, components: dict[ComponentType, TreeComponent]
) -> UnifiedTreeCoreInterface:
    """Load and construct the in-tree Rust TreeCore only when selected."""
    from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore

    return RustUnifiedTreeCore(params)


register_tree_core_backend("python", _python_tree_core_factory)
register_tree_core_backend("rust", _rust_tree_core_factory)


def create_tree_core(
    name: str,
    params: CacheInitParams,
    components: dict[ComponentType, TreeComponent],
) -> UnifiedTreeCoreInterface:
    """Construct the TreeCore registered under `name`."""
    factory = get_tree_core_factory(name)
    if factory is None:
        raise ValueError(
            f"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND={name!r} is not registered. "
            f"Registered backends: {registered_tree_core_backends()}. "
            "External backends must call register_tree_core_backend(...) at import time."
        )
    return factory(params, components)
