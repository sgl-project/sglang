"""Names and component kinds implemented by the native Rust registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

_RUST_TREE_COMPONENT_REGISTRY: dict[str, ComponentType] = {}


def register_rust_tree_component(name: str, component_type: ComponentType) -> None:
    """Declare the name and kind of a component compiled into the Rust registry."""
    if not name.strip():
        raise ValueError("Rust component name must be non-empty")
    if not isinstance(component_type, ComponentType):
        raise TypeError("Rust components must declare a ComponentType")
    existing = _RUST_TREE_COMPONENT_REGISTRY.get(name)
    if existing is not None and existing != component_type:
        raise ValueError(f"Rust component {name!r} is already registered")
    _RUST_TREE_COMPONENT_REGISTRY[name] = component_type


def get_rust_tree_component(name: str) -> ComponentType | None:
    return _RUST_TREE_COMPONENT_REGISTRY.get(name)


def registered_rust_tree_components() -> dict[str, ComponentType]:
    return dict(_RUST_TREE_COMPONENT_REGISTRY)


def resolve_rust_component_overrides(params: CacheInitParams) -> list[tuple[int, str]]:
    """Resolve named overrides without loading the native extension."""
    result = []
    for component_type, name in (params.component_registry_override or {}).items():
        if component_type not in (params.tree_components or ()):
            raise ValueError(
                f"component_registry_override targets inactive {component_type}"
            )
        if not isinstance(name, str) or get_rust_tree_component(name) != component_type:
            raise ValueError(
                f"Rust TreeCore does not support component_registry_override "
                f"{name!r} for {component_type.name}"
            )
        result.append((int(component_type), name))
    return result


for _component_type in (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA):
    register_rust_tree_component(_component_type.name.lower(), _component_type)
