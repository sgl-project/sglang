"""Named Python component implementations for the unified cache."""

from __future__ import annotations

from typing import Callable

from sglang.srt.mem_cache.unified_cache.components.base import (
    PythonTreeComponentArgument,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent

PythonTreeComponentFactory = (
    type[TreeComponent] | Callable[[PythonTreeComponentArgument], TreeComponent]
)
_PYTHON_TREE_COMPONENT_REGISTRY: dict[str, PythonTreeComponentFactory] = {}


def register_python_tree_component(
    name: str, factory: PythonTreeComponentFactory, replace: bool = False
) -> None:
    """Register a component factory for subsequently constructed caches."""
    if not name.strip():
        raise ValueError("Python component name must be non-empty")
    if not callable(factory):
        raise TypeError("Python component factories must be callable")
    existing = _PYTHON_TREE_COMPONENT_REGISTRY.get(name)
    if existing is not None and existing is not factory and not replace:
        raise ValueError(f"Python component {name!r} is already registered")
    _PYTHON_TREE_COMPONENT_REGISTRY[name] = factory


def get_python_tree_component(name: str) -> PythonTreeComponentFactory | None:
    return _PYTHON_TREE_COMPONENT_REGISTRY.get(name)


def registered_python_tree_components() -> dict[str, PythonTreeComponentFactory]:
    return dict(_PYTHON_TREE_COMPONENT_REGISTRY)


def create_python_tree_component(
    factory: PythonTreeComponentFactory, args: PythonTreeComponentArgument
) -> TreeComponent:
    """Construct and validate a Python component from a class or factory."""
    if isinstance(factory, type) and issubclass(factory, TreeComponent):
        if factory.component_type != args.component_type:
            raise ValueError(
                f"Python component factory has kind {factory.component_type.name}, "
                f"expected {args.component_type.name}"
            )
        component = factory(args.cache, args.params)
    else:
        component = factory(args)
    if not isinstance(component, TreeComponent):
        raise TypeError("Python component factories must return a TreeComponent")
    if component.component_type != args.component_type:
        raise ValueError(
            f"Python component factory returned {component.component_type.name}, "
            f"expected {args.component_type.name}"
        )
    return component


register_python_tree_component("full", FullComponent)
register_python_tree_component("swa", SWAComponent)
register_python_tree_component("mamba", MambaComponent)
