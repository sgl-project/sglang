"""Factory keys and selectors shared by component registries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.components.base import TreeComponent

DEFAULT_COMPONENT_FACTORY_KEYS: dict[ComponentType, str] = {
    ComponentType.FULL: "full",
    ComponentType.SWA: "swa",
    ComponentType.MAMBA: "mamba",
}


def resolve_component_factory_keys(
    params: CacheInitParams,
) -> dict[ComponentType, str | type[TreeComponent]]:
    """Resolve component selectors while preserving legacy Python class inputs."""
    overrides = params.component_registry_override or {}
    active = params.tree_components or ()
    for component_type, selector in overrides.items():
        if not isinstance(selector, type) and component_type not in active:
            raise ValueError(
                f"component_registry_override targets inactive {component_type}"
            )
    result = {}
    for component_type in active:
        selector = overrides.get(
            component_type,
            DEFAULT_COMPONENT_FACTORY_KEYS.get(component_type, component_type),
        )
        if isinstance(selector, ComponentType):
            if selector != component_type:
                raise ValueError(
                    f"component_registry_override cannot use {selector.name} "
                    f"for {component_type.name}"
                )
            if selector not in DEFAULT_COMPONENT_FACTORY_KEYS:
                raise ValueError(f"No default component factory for {selector.name}")
            selector = DEFAULT_COMPONENT_FACTORY_KEYS[selector]
        if isinstance(selector, str):
            if not selector.strip():
                raise ValueError("Component factory key must be non-empty")
        elif not isinstance(selector, type):
            raise TypeError(
                "component_registry_override requires a factory name, "
                "ComponentType, or Python component class"
            )
        result[component_type] = selector
    return result
