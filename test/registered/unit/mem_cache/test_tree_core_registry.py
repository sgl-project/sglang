"""Unit tests for the tree-core backend registry."""

import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    EvictLayer,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
    _TREE_CORE_REGISTRY,
    cpp_tree_core_unsupported_reason,
    create_tree_core,
    register_tree_core_backend,
    registered_tree_core_backends,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cache_init_params(
    tree_components=(ComponentType.FULL,), **kwargs
) -> CacheInitParams:
    return CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=2,
        tree_components=tree_components,
        **kwargs,
    )


class _StubFullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def create_match_validator(self, match_device_only: bool = False):
        return lambda node: True

    def redistribute_on_node_split(self, new_parent, child):
        return None

    def evict_component(
        self, node, device_frees, host_frees, target: EvictLayer = EvictLayer.DEVICE
    ) -> tuple[int, int]:
        return 0, 0

    def acquire_component_lock(self, node, result):
        return result

    def release_component_lock(self, node, params):
        return None

    def _evict_device_start(self, request_cnt) -> None:
        pass

    def _evict_device_next_node(self, tracker, device_frees, host_frees):
        return None

    def _evict_device_end(self) -> None:
        pass

    def _dec_session_coverage(self, session_id, leaf) -> None:
        pass

    def _advance_session_coverage(self, session_id, leaf, old_ancestor) -> None:
        pass

    def _recede_session_coverage(self, session_id, leaf, fallback) -> None:
        pass


class _StubMambaComponent(_StubFullComponent):
    component_type = ComponentType.MAMBA


class TreeCoreRegistryTest(CustomTestCase):
    def setUp(self):
        self._registry_snapshot = dict(_TREE_CORE_REGISTRY)

    def tearDown(self):
        _TREE_CORE_REGISTRY.clear()
        _TREE_CORE_REGISTRY.update(self._registry_snapshot)

    def test_registry_contains_python(self):
        self.assertIn("python", registered_tree_core_backends())

    def test_registry_contains_cpp(self):
        self.assertIn("cpp", registered_tree_core_backends())

    def test_registry_contains_auto(self):
        self.assertIn("auto", registered_tree_core_backends())

    def test_python_backend_builds_the_python_tree(self):
        component = mock.MagicMock()
        core = create_tree_core(
            name="python",
            params=_cache_init_params(),
            components={ComponentType.FULL: component},
        )
        self.assertIsInstance(core, UnifiedTreeCore)
        self.assertIs(component.tree_core, core)

    def test_unknown_backend_raises_naming_the_known_backends(self):
        with self.assertRaisesRegex(ValueError, "not registered") as cm:
            create_tree_core(
                name="not_a_real_backend",
                params=_cache_init_params(),
                components={},
            )
        self.assertIn("'python'", str(cm.exception))

    def test_register_rejects_empty_name(self):
        with self.assertRaises(ValueError):
            register_tree_core_backend("   ", mock.MagicMock())

    def test_register_rejects_duplicate_name(self):
        with self.assertRaises(ValueError):
            register_tree_core_backend("python", mock.MagicMock())

    def test_create_dispatches_to_a_registered_factory(self):
        core = mock.MagicMock()
        factory = mock.MagicMock(return_value=core)
        register_tree_core_backend("custom", factory)
        params = _cache_init_params()
        components = {ComponentType.FULL: mock.MagicMock()}
        result = create_tree_core(name="custom", params=params, components=components)
        factory.assert_called_once_with(params, components)
        self.assertIs(result, core)

    def test_auto_selects_cpp_for_full_device_only(self):
        params = _cache_init_params()
        components = {ComponentType.FULL: mock.MagicMock()}
        cpp_core = mock.MagicMock()
        with mock.patch(
            "sglang.srt.mem_cache.unified_cache.tree_core_registry._cpp_tree_core_factory",
            return_value=cpp_core,
        ) as cpp_factory:
            result = create_tree_core("auto", params, components)
        cpp_factory.assert_called_once_with(params, components)
        self.assertIs(result, cpp_core)

    def test_auto_selects_cpp_for_full_swa_device_only(self):
        params = _cache_init_params(
            tree_components=(ComponentType.FULL, ComponentType.SWA),
            sliding_window_size=4,
        )
        components = {
            ComponentType.FULL: mock.MagicMock(),
            ComponentType.SWA: mock.MagicMock(),
        }
        cpp_core = mock.MagicMock()
        with mock.patch(
            "sglang.srt.mem_cache.unified_cache.tree_core_registry._cpp_tree_core_factory",
            return_value=cpp_core,
        ) as cpp_factory:
            result = create_tree_core("auto", params, components)
        cpp_factory.assert_called_once_with(params, components)
        self.assertIs(result, cpp_core)

    def test_cpp_compatibility_rejects_swa_without_window_size(self):
        params = _cache_init_params(
            tree_components=(ComponentType.FULL, ComponentType.SWA)
        )
        reason = cpp_tree_core_unsupported_reason(
            params,
            {
                ComponentType.FULL: mock.MagicMock(),
                ComponentType.SWA: mock.MagicMock(),
            },
        )
        self.assertIn("sliding_window_size", reason)

    def test_auto_falls_back_to_python_for_unsupported_components(self):
        params = _cache_init_params(
            tree_components=(ComponentType.FULL, ComponentType.MAMBA)
        )
        components = {
            ComponentType.FULL: mock.MagicMock(),
            ComponentType.MAMBA: mock.MagicMock(),
        }
        python_core = mock.MagicMock()
        with (
            mock.patch(
                "sglang.srt.mem_cache.unified_cache.tree_core_registry._python_tree_core_factory",
                return_value=python_core,
            ) as python_factory,
            mock.patch(
                "sglang.srt.mem_cache.unified_cache.tree_core_registry._cpp_tree_core_factory"
            ) as cpp_factory,
        ):
            result = create_tree_core("auto", params, components)
        python_factory.assert_called_once_with(params, components)
        cpp_factory.assert_not_called()
        self.assertIs(result, python_core)

    def test_cpp_compatibility_rejects_hicache(self):
        params = _cache_init_params(enable_hicache=True)
        reason = cpp_tree_core_unsupported_reason(
            params, {ComponentType.FULL: mock.MagicMock()}
        )
        self.assertIn("HiCache", reason)


class UnifiedRadixCacheTreeCoreSelectionTest(CustomTestCase):
    def setUp(self):
        self._registry_snapshot = dict(_TREE_CORE_REGISTRY)

    def tearDown(self):
        _TREE_CORE_REGISTRY.clear()
        _TREE_CORE_REGISTRY.update(self._registry_snapshot)

    def _cache_params(
        self,
        tree_components=(ComponentType.FULL,),
        component_registry_override={ComponentType.FULL: _StubFullComponent},
        **kwargs,
    ) -> CacheInitParams:
        return CacheInitParams(
            disable=True,
            req_to_token_pool=ReqToTokenPool(
                size=2,
                max_context_len=8,
                device="cpu",
                enable_memory_saver=False,
            ),
            token_to_kv_pool_allocator=None,
            page_size=1,
            tree_components=tree_components,
            component_registry_override=component_registry_override,
            **kwargs,
        )

    def test_init_downgrades_is_eagle_when_mamba_is_enabled(self):
        params = self._cache_params(
            is_eagle=True,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            component_registry_override={
                ComponentType.FULL: _StubFullComponent,
                ComponentType.MAMBA: _StubMambaComponent,
            },
        )
        cache = UnifiedRadixCache(params)
        self.assertFalse(cache.tree_core.is_eagle)

    def test_init_keeps_is_eagle_without_mamba(self):
        params = self._cache_params(is_eagle=True)
        cache = UnifiedRadixCache(params)
        self.assertTrue(cache.tree_core.is_eagle)

    def test_default_backend_routes_to_auto(self):
        core = mock.MagicMock()
        factory = mock.MagicMock(return_value=core)
        _TREE_CORE_REGISTRY["auto"] = factory
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("auto"):
            cache = UnifiedRadixCache(params=self._cache_params())
        factory.assert_called_once()
        self.assertIs(cache.tree_core, core)
        component = cache.components[ComponentType.FULL]
        self.assertIs(component.tree_core, core)

    def test_env_var_routes_construction_to_the_selected_backend(self):
        """SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND selects the registered factory
        the cache constructs its tree through."""
        core = mock.MagicMock()
        factory = mock.MagicMock(return_value=core)
        register_tree_core_backend("custom_env_backend", factory)
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("custom_env_backend"):
            cache = UnifiedRadixCache(params=self._cache_params())
        factory.assert_called_once()
        self.assertIs(cache.tree_core, core)
        component = cache.components[ComponentType.FULL]
        self.assertIs(component.tree_core, core)


if __name__ == "__main__":
    unittest.main()
