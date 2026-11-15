"""Use the shared TreeCore inspectors in scripted-runtime test processes."""


def install_tree_core_inspectors():
    from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
        _TREE_CORE_REGISTRY,
    )

    def python_inspector_factory(params, components):
        from sglang.test.kits.unified_radix_cache.unified_tree_core_inspector import (
            UnifiedTreeCoreInspector,
        )

        return UnifiedTreeCoreInspector(params, components)

    def rust_inspector_factory(params, _components):
        from sglang.test.kits.unified_radix_cache.rust_unified_tree_core_inspector import (
            RustUnifiedTreeCoreInspector,
        )

        return RustUnifiedTreeCoreInspector(params)

    _TREE_CORE_REGISTRY.update(
        python=python_inspector_factory, rust=rust_inspector_factory
    )
