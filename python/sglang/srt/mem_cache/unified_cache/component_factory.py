"""Factory keys and selectors shared by component registries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.components.base import TreeComponent

DEFAULT_COMPONENT_FACTORY_KEYS: dict[ComponentType, str] = {
    ComponentType.FULL: "full_default",
    ComponentType.SWA: "swa_default",
    ComponentType.MAMBA: "mamba_default",
}


def resolve_component_factory_keys(
    params: CacheInitParams,
) -> dict[ComponentType, str | type[TreeComponent]]:
    """Merge factory keys while preserving legacy Python class inputs."""
    overrides = params.component_registry_override or {}
    active = params.tree_components or ()
    for component_type, selector in overrides.items():
        if isinstance(selector, str):
            if not selector.strip():
                raise ValueError("Component factory key must be non-empty")
            if component_type not in active:
                raise ValueError(
                    f"component_registry_override targets inactive {component_type}"
                )
        elif not isinstance(selector, type):
            raise TypeError(
                "component_registry_override requires a factory name "
                "or Python component class"
            )
    factory_keys: dict[ComponentType, str | type[TreeComponent]] = dict(
        DEFAULT_COMPONENT_FACTORY_KEYS
    )
    factory_keys.update(overrides)
    for component_type in active:
        if component_type not in factory_keys:
            raise ValueError(f"No default component factory for {component_type.name}")
    return {component_type: factory_keys[component_type] for component_type in active}
