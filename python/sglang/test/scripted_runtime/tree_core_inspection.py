"""Use the shared TreeCore inspectors in scripted-runtime test processes."""

import sys
from pathlib import Path


def install_tree_core_inspectors():
    from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
        _TREE_CORE_REGISTRY,
    )

    inspector_dir = str(
        Path(__file__).resolve().parents[4]
        / "test"
        / "registered"
        / "unit"
        / "mem_cache"
    )
    if inspector_dir not in sys.path:
        sys.path.insert(0, inspector_dir)

    def python_inspector_factory(params, components):
        from unified_tree_core_inspector import UnifiedTreeCoreInspector

        return UnifiedTreeCoreInspector(params, components)

    def rust_inspector_factory(params, _components):
        from rust_unified_tree_core_inspector import RustUnifiedTreeCoreInspector

        return RustUnifiedTreeCoreInspector(params)

    _TREE_CORE_REGISTRY.update(
        python=python_inspector_factory, rust=rust_inspector_factory
    )
