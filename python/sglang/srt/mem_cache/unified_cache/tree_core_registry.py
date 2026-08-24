"""Registry for pluggable TreeCore implementations.

The unified cache constructs its TreeCore through `create_tree_core`, selected
by SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND (default "auto"). To plug in a custom
implementation, register it under a string name via
`register_tree_core_backend(name, factory)`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.components import TreeComponent
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
        UnifiedTreeCoreInterface,
    )

TreeCoreFactory = Callable[
    ["CacheInitParams", "dict[ComponentType, TreeComponent]"],
    "UnifiedTreeCoreInterface",
]

_TREE_CORE_REGISTRY: dict[str, TreeCoreFactory] = {}

logger = logging.getLogger(__name__)


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


def _cpp_tree_core_factory(
    params: CacheInitParams, components: dict[ComponentType, TreeComponent]
) -> UnifiedTreeCoreInterface:
    """The experimental C++ FULL or FULL+SWA device-only TreeCore.

    Keep the extension import lazy: selecting the Python backend must not pay
    the C++ JIT build cost.
    """
    from sglang.srt.mem_cache.unified_cache.cpp_unified_tree_core import (
        CppUnifiedTreeCore,
    )

    return CppUnifiedTreeCore(params, components)


def cpp_tree_core_unsupported_reason(
    params: CacheInitParams, components: dict[ComponentType, TreeComponent]
) -> Optional[str]:
    """Return why the first-stage C++ core cannot serve this configuration."""
    component_types = tuple(components)
    supported_components = (
        (ComponentType.FULL,),
        (ComponentType.FULL, ComponentType.SWA),
    )
    if component_types not in supported_components:
        return (
            f"components={component_types!r} "
            "(only FULL and device-only FULL+SWA are supported)"
        )
    if ComponentType.SWA in component_types and not params.sliding_window_size:
        return "SWA component has no sliding_window_size"
    if params.enable_hicache:
        return "HiCache/host-tier caching is enabled"
    if params.enable_session_radix_cache:
        return "session radix cache is enabled"
    if params.enable_kv_cache_events:
        return "KV cache events are enabled"
    if params.is_eagle:
        return "EAGLE/bigram keys are enabled"
    if params.eviction_policy.lower() != "lru":
        return f"eviction policy is {params.eviction_policy!r} (only LRU is supported)"
    return None


def _auto_tree_core_factory(
    params: CacheInitParams, components: dict[ComponentType, TreeComponent]
) -> UnifiedTreeCoreInterface:
    """Prefer C++ when its current feature envelope covers the configuration."""
    reason = cpp_tree_core_unsupported_reason(params, components)
    if reason is None:
        return _cpp_tree_core_factory(params, components)

    logger.info(
        "Using Python unified radix TreeCore because the C++ backend does not "
        "support this configuration yet: %s.",
        reason,
    )
    return _python_tree_core_factory(params, components)


register_tree_core_backend("python", _python_tree_core_factory)
register_tree_core_backend("cpp", _cpp_tree_core_factory)
register_tree_core_backend("auto", _auto_tree_core_factory)


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
