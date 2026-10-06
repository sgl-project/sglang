"""Named Python component implementations for the unified cache."""

from __future__ import annotations

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import TreeComponent
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent

COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
    ComponentType.MAMBA: MambaComponent,
    ComponentType.SWA: SWAComponent,
}
_PYTHON_TREE_COMPONENT_REGISTRY: dict[str, type[TreeComponent]] = {}


def register_python_tree_component(name: str, component: type[TreeComponent]) -> None:
    """Register a Python component under a backend-independent name."""
    if not name.strip():
        raise ValueError("Python component name must be non-empty")
    if not isinstance(component, type) or not issubclass(component, TreeComponent):
        raise TypeError("Python components must inherit TreeComponent")
    if not isinstance(getattr(component, "component_type", None), ComponentType):
        raise TypeError("Python components must declare their component_type")
    existing = _PYTHON_TREE_COMPONENT_REGISTRY.get(name)
    if existing is not None and existing is not component:
        raise ValueError(f"Python component {name!r} is already registered")
    _PYTHON_TREE_COMPONENT_REGISTRY[name] = component


def get_python_tree_component(name: str) -> type[TreeComponent] | None:
    return _PYTHON_TREE_COMPONENT_REGISTRY.get(name)


def registered_python_tree_components() -> dict[str, type[TreeComponent]]:
    return dict(_PYTHON_TREE_COMPONENT_REGISTRY)


for _component_type, _component in COMPONENT_REGISTRY.items():
    register_python_tree_component(_component_type.name.lower(), _component)
