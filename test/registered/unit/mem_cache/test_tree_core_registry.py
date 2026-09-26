"""Unit tests for the tree-core backend registry."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from unified_tree_core_inspection_interface import UnifiedTreeCoreInspectionInterface

from sglang.srt.environ import envs
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache import tree_core_registry
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import EvictLayer, TreeComponent
from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
    _TREE_CORE_REGISTRY,
    create_tree_core,
    register_tree_core_backend,
    registered_tree_core_backends,
    resolve_tree_core_backend,
    select_tree_core_backend,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _cache_init_params(**kwargs) -> CacheInitParams:
    return CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=2,
        tree_components=(ComponentType.FULL,),
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


class _StubAuxiliarySWAComponent(_StubFullComponent):
    component_type = ComponentType.AUXILIARY_SWA


class TreeCoreRegistryTest(CustomTestCase):
    def setUp(self):
        self._registry_snapshot = dict(_TREE_CORE_REGISTRY)

    def tearDown(self):
        _TREE_CORE_REGISTRY.clear()
        _TREE_CORE_REGISTRY.update(self._registry_snapshot)

    def test_registry_contains_python(self):
        self.assertIn("python", registered_tree_core_backends())

    def test_python_backend_builds_the_python_tree(self):
        component = mock.MagicMock()
        core = create_tree_core(
            name="python",
            params=_cache_init_params(),
            components={ComponentType.FULL: component},
        )
        self.assertIsInstance(core, UnifiedTreeCore)
        self.assertNotIsInstance(core, UnifiedTreeCoreInspectionInterface)
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

    def test_custom_components_default_to_the_python_tree_core(self):
        cache = UnifiedRadixCache(params=self._cache_params())
        self.assertIsInstance(cache.tree_core, UnifiedTreeCore)
        component = cache.components[ComponentType.FULL]
        self.assertIs(component.tree_core, cache.tree_core)

    def test_explicit_rust_cache_records_the_python_fallback(self):
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("rust"):
            cache = UnifiedRadixCache(params=self._cache_params())
        self.assertIsInstance(cache.tree_core, UnifiedTreeCore)
        self.assertEqual(cache._tree_core_backend, "python")

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

    def test_backend_override_is_instance_local(self):
        for backend, default_backend, expected_backend in (
            ("python", "unregistered-test-core", "python"),
            (None, "python", "python"),
            (None, "unregistered-test-core", None),
            ("", "python", None),
            ("unregistered-test-core", "python", None),
        ):
            with (
                self.subTest(backend=backend, default_backend=default_backend),
                envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(default_backend),
            ):
                params = self._cache_params(tree_core_backend=backend)
                if expected_backend is None:
                    with self.assertRaisesRegex(ValueError, "is not registered"):
                        UnifiedRadixCache(params)
                else:
                    cache = UnifiedRadixCache(params)
                    self.assertIsInstance(cache.tree_core, UnifiedTreeCore)
                    self.assertEqual(cache._tree_core_backend, expected_backend)
                self.assertEqual(
                    envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get(), default_backend
                )

    def test_auxiliary_component_uses_its_own_node_storage(self):
        cache = UnifiedRadixCache(
            self._cache_params(
                tree_components=(ComponentType.FULL, ComponentType.AUXILIARY_SWA),
                component_registry_override={
                    ComponentType.FULL: _StubFullComponent,
                    ComponentType.AUXILIARY_SWA: _StubAuxiliarySWAComponent,
                },
                tree_core_backend="python",
            )
        )
        node = cache.tree_core.root_node
        auxiliary = node.component_data[ComponentType.AUXILIARY_SWA]
        full = node.component_data[ComponentType.FULL]
        self.assertIsNot(auxiliary, full)
        auxiliary.metadata["boundary"] = 7
        self.assertNotIn("boundary", full.metadata)
        self.assertIs(
            cache.components[ComponentType.AUXILIARY_SWA].tree_core, cache.tree_core
        )

    def test_supported_default_routes_to_rust(self):
        core = mock.MagicMock()
        factory = mock.MagicMock(return_value=core)
        params = self._cache_params(component_registry_override=None)
        with (
            mock.patch.dict(os.environ),
            mock.patch.object(
                tree_core_registry,
                "_rust_fallback_reason",
                return_value=None,
            ),
            mock.patch.dict(_TREE_CORE_REGISTRY, {"rust": factory}),
        ):
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.clear()
            cache = UnifiedRadixCache(params)
        factory.assert_called_once_with(params, cache.components)
        self.assertIs(cache.tree_core, core)

    def test_default_rust_construction_errors_propagate(self):
        factory = mock.MagicMock(side_effect=RuntimeError("extension build failed"))
        with (
            mock.patch.dict(os.environ),
            mock.patch.dict(tree_core_registry.sys.modules),
            mock.patch.object(tree_core_registry.sys, "platform", "linux"),
            mock.patch.object(tree_core_registry.torch, "__version__", "2.13.0"),
            mock.patch.object(
                tree_core_registry.importlib.util,
                "find_spec",
                return_value=None,
            ),
            mock.patch.object(
                tree_core_registry.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, stdout="version"),
            ) as run,
            mock.patch.dict(_TREE_CORE_REGISTRY, {"rust": factory}),
            envs.SGLANG_RUST_BUILD_MODE.override("auto"),
        ):
            tree_core_registry.sys.modules.pop(
                tree_core_registry._RUST_TREE_CORE_MODULE, None
            )
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.clear()
            with self.assertRaisesRegex(RuntimeError, "extension build failed"):
                UnifiedRadixCache(self._cache_params(component_registry_override=None))
        self.assertTrue(run.called)

    def test_default_cache_falls_back_for_unavailable_toolchain(self):
        params = self._cache_params(component_registry_override=None)
        for unavailable in ("missing", "cargo", "rustc"):
            with (
                self.subTest(unavailable=unavailable),
                tempfile.TemporaryDirectory() as tmp,
            ):
                if unavailable != "missing":
                    for command in ("cargo", "rustc"):
                        executable = Path(tmp) / command
                        executable.write_text(
                            "#!/bin/sh\n"
                            + ("exit 1\n" if command == unavailable else "exit 0\n")
                        )
                        executable.chmod(0o755)
                rust_factory = mock.MagicMock(
                    side_effect=AssertionError(
                        "unavailable Rust toolchain was selected"
                    )
                )
                with (
                    mock.patch.dict(os.environ, {"PATH": tmp}),
                    mock.patch.dict(tree_core_registry.sys.modules),
                    mock.patch.object(tree_core_registry.sys, "platform", "linux"),
                    mock.patch.object(
                        tree_core_registry.torch, "__version__", "2.13.0"
                    ),
                    mock.patch.object(
                        tree_core_registry.importlib.util,
                        "find_spec",
                        return_value=None,
                    ),
                    mock.patch.dict(_TREE_CORE_REGISTRY, {"rust": rust_factory}),
                    envs.SGLANG_RUST_BUILD_MODE.override("auto"),
                ):
                    tree_core_registry.sys.modules.pop(
                        tree_core_registry._RUST_TREE_CORE_MODULE, None
                    )
                    envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.clear()
                    cache = UnifiedRadixCache(params)
                self.assertIsInstance(cache.tree_core, UnifiedTreeCore)
                self.assertEqual(cache._tree_core_backend, "python")
                rust_factory.assert_not_called()


class TreeCoreDefaultCompatibilityTest(CustomTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        (self.workspace / "Cargo.toml").touch()
        patchers = (
            mock.patch.dict(os.environ, {"SGLANG_RUST_BUILD_MODE": "auto"}),
            mock.patch.dict(tree_core_registry.sys.modules),
            mock.patch.object(tree_core_registry.sys, "platform", "linux"),
            mock.patch.object(tree_core_registry.torch, "__version__", "2.13.0"),
            mock.patch.object(
                tree_core_registry.importlib.util, "find_spec", return_value=None
            ),
            mock.patch.object(tree_core_registry, "_RUST_TREE_CORE_MANIFEST"),
            mock.patch.object(
                tree_core_registry.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, stdout="version"),
            ),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.clear()
        tree_core_registry.sys.modules.pop(
            tree_core_registry._RUST_TREE_CORE_MODULE, None
        )
        tree_core_registry._RUST_TREE_CORE_MANIFEST.is_file.return_value = True
        tree_core_registry._RUST_TREE_CORE_MANIFEST.parent.parent = self.workspace

    def test_source_install_defaults_to_rust_on_cpu_and_cuda(self):
        for device in ("cpu", "cuda:1"):
            with self.subTest(device=device):
                params = _cache_init_params()
                params.token_to_kv_pool_allocator = SimpleNamespace(device=device)
                self.assertEqual(select_tree_core_backend(params), "rust")

    def test_bundled_extension_needs_neither_sources_nor_toolchain(self):
        (self.workspace / "Cargo.toml").unlink()
        tree_core_registry._RUST_TREE_CORE_MANIFEST.is_file.return_value = False
        tree_core_registry.importlib.util.find_spec.return_value = object()
        self.assertEqual(select_tree_core_backend(_cache_init_params()), "rust")
        tree_core_registry.subprocess.run.assert_not_called()

    def test_loaded_extension_needs_no_toolchain(self):
        tree_core_registry.sys.modules[tree_core_registry._RUST_TREE_CORE_MODULE] = (
            SimpleNamespace()
        )
        self.assertEqual(select_tree_core_backend(_cache_init_params()), "rust")
        tree_core_registry.subprocess.run.assert_not_called()

    def test_never_mode_trusts_bundled_extension_in_a_source_checkout(self):
        tree_core_registry.importlib.util.find_spec.return_value = object()
        with envs.SGLANG_RUST_BUILD_MODE.override("never"):
            self.assertEqual(select_tree_core_backend(_cache_init_params()), "rust")
        tree_core_registry.subprocess.run.assert_not_called()

    def test_source_build_modes_probe_tools_even_with_a_bundled_extension(self):
        tree_core_registry.importlib.util.find_spec.return_value = object()
        for mode in ("auto", "force"):
            with self.subTest(mode=mode), envs.SGLANG_RUST_BUILD_MODE.override(mode):
                self.assertEqual(select_tree_core_backend(_cache_init_params()), "rust")
                self.assertEqual(
                    tree_core_registry.subprocess.run.call_args_list,
                    [
                        mock.call(
                            command,
                            cwd=self.workspace,
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        for command in (
                            ("cargo", "--version", "--verbose"),
                            ("rustc", "-vV"),
                        )
                    ],
                )
                tree_core_registry.subprocess.run.reset_mock()

    def test_toolchain_probe_failure_uses_python_for_all_source_build_modes(self):
        for mode in ("auto", "never", "force"):
            for error in (
                FileNotFoundError("cargo"),
                subprocess.CalledProcessError(1, "cargo"),
                subprocess.TimeoutExpired("rustc", 10),
            ):
                with (
                    self.subTest(mode=mode, error=type(error).__name__),
                    envs.SGLANG_RUST_BUILD_MODE.override(mode),
                ):
                    tree_core_registry.subprocess.run.side_effect = error
                    self.assertEqual(
                        select_tree_core_backend(_cache_init_params()), "python"
                    )
                    self.assertEqual(
                        resolve_tree_core_backend("rust", _cache_init_params()),
                        "python",
                    )

    def test_invalid_mode_and_force_after_import_remain_loader_errors(self):
        from sglang.srt.rust_extensions.loader import load_rust_extension

        module = tree_core_registry._RUST_TREE_CORE_MODULE
        for mode, loaded, error, message in (
            ("invalid", False, ValueError, "invalid Rust extension build mode"),
            ("force", True, RuntimeError, "cannot force-build"),
        ):
            with (
                self.subTest(mode=mode),
                envs.SGLANG_RUST_BUILD_MODE.override(mode),
                mock.patch.dict(
                    _TREE_CORE_REGISTRY,
                    {"rust": lambda params, components: load_rust_extension(module)},
                ),
            ):
                if loaded:
                    tree_core_registry.sys.modules[module] = SimpleNamespace()
                with self.assertRaisesRegex(error, message):
                    create_tree_core("rust", _cache_init_params(), {})
        tree_core_registry.subprocess.run.assert_not_called()

    def test_platform_distribution_without_tree_core_uses_python(self):
        tree_core_registry._RUST_TREE_CORE_MANIFEST.is_file.return_value = False
        self.assertEqual(select_tree_core_backend(_cache_init_params()), "python")

    def test_session_and_custom_components_use_python(self):
        for overrides in (
            {"enable_session_radix_cache": True},
            {"component_registry_override": {ComponentType.FULL: _StubFullComponent}},
        ):
            with self.subTest(overrides=overrides):
                self.assertEqual(
                    select_tree_core_backend(_cache_init_params(**overrides)), "python"
                )
        params = _cache_init_params()
        params.tree_components = (ComponentType.FULL, ComponentType.C128)
        self.assertEqual(select_tree_core_backend(params), "python")

    def test_unsupported_platform_and_torch_use_python(self):
        for platform in ("darwin", "win32"):
            with (
                self.subTest(platform=platform),
                mock.patch.object(tree_core_registry.sys, "platform", platform),
            ):
                self.assertEqual(
                    select_tree_core_backend(_cache_init_params()), "python"
                )
        for version in ("2.10.0", "2.14.0", "unknown"):
            with (
                self.subTest(version=version),
                mock.patch.object(tree_core_registry.torch, "__version__", version),
            ):
                self.assertEqual(
                    select_tree_core_backend(_cache_init_params()), "python"
                )

    def test_unsupported_device_uses_python(self):
        params = _cache_init_params()
        params.token_to_kv_pool_allocator = SimpleNamespace(device="xpu:0")
        self.assertEqual(select_tree_core_backend(params), "python")

    def test_explicit_rust_uses_the_same_compatibility_fallback(self):
        params = _cache_init_params(enable_session_radix_cache=True)
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("rust"):
            self.assertEqual(select_tree_core_backend(params), "python")
        self.assertEqual(resolve_tree_core_backend("rust", params), "python")

    def test_instance_rust_override_uses_the_same_compatibility_fallback(self):
        for session_enabled, expected in ((False, "rust"), (True, "python")):
            with (
                self.subTest(session_enabled=session_enabled),
                envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"),
            ):
                params = _cache_init_params(
                    tree_core_backend="rust",
                    enable_session_radix_cache=session_enabled,
                )
                self.assertEqual(select_tree_core_backend(params), expected)
                self.assertEqual(
                    envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get(), "python"
                )

    def test_numeric_tlru_uses_rust_and_preserves_policy_configuration(self):
        for config in (
            {"threshold": 4096, "next_prompt_estimate": 1024},
            {"threshold": 2**100 + 4, "next_prompt_estimate": 2**100},
            {"threshold": 4096.0, "next_prompt_estimate": 1024.0},
            {"threshold": 4096.5, "next_prompt_estimate": 1024.25},
            {"threshold": 1.2, "next_prompt_estimate": 0.2},
            {"threshold": float(2**53 + 2), "next_prompt_estimate": 2**53 + 1},
            {"threshold": -0.5, "next_prompt_estimate": -2},
            {"threshold": float("inf"), "next_prompt_estimate": 0},
            {"threshold": 0, "next_prompt_estimate": float("nan")},
        ):
            with self.subTest(config=config):
                params = _cache_init_params(
                    eviction_policy="TLRU", eviction_policy_config=config
                )
                self.assertEqual(select_tree_core_backend(params), "rust")
                with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("rust"):
                    self.assertEqual(select_tree_core_backend(params), "rust")
                components = {ComponentType.FULL: mock.MagicMock()}
                factory = mock.MagicMock()
                with mock.patch.dict(_TREE_CORE_REGISTRY, {"rust": factory}):
                    core = create_tree_core("rust", params, components)
                factory.assert_called_once_with(params, components)
                self.assertIs(core, factory.return_value)
                self.assertEqual(params.eviction_policy_config, config)

    def test_python_and_custom_backend_selections_are_unchanged(self):
        params = _cache_init_params(enable_session_radix_cache=True)
        for backend in ("python", "custom_backend"):
            with (
                self.subTest(backend=backend),
                envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(backend),
            ):
                self.assertEqual(select_tree_core_backend(params), backend)

    def test_factory_resolves_rust_fallback_before_loading_the_extension(self):
        params = _cache_init_params(enable_session_radix_cache=True)
        components = {ComponentType.FULL: mock.MagicMock()}
        python_factory = mock.MagicMock()
        rust_factory = mock.MagicMock(side_effect=AssertionError("Rust was loaded"))
        with mock.patch.dict(
            _TREE_CORE_REGISTRY, {"python": python_factory, "rust": rust_factory}
        ):
            result = create_tree_core("rust", params, components)
        python_factory.assert_called_once_with(params, components)
        rust_factory.assert_not_called()
        self.assertIs(result, python_factory.return_value)

    def test_supported_explicit_rust_factory_errors_propagate(self):
        rust_factory = mock.MagicMock(side_effect=RuntimeError("extension load failed"))
        python_factory = mock.MagicMock()
        with mock.patch.dict(
            _TREE_CORE_REGISTRY, {"python": python_factory, "rust": rust_factory}
        ):
            with self.assertRaisesRegex(RuntimeError, "extension load failed"):
                create_tree_core("rust", _cache_init_params(), {})
        python_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
