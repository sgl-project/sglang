"""Configuration and real Python-module loading for external linkers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import argparse
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.hicache_hook import handle_hicache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.unified_cache.linker_config import UnifiedCacheLinkerConfig
from sglang.srt.mem_cache.unified_cache.linker_factory import (
    create_unified_cache_linker,
    get_startup_linker_config,
)
from sglang.srt.runtime_context import get_context, get_memory
from sglang.srt.server_args import ServerArgs

_CONFIG = {
    "linker": "ExternalLinker",
    "linker_module_path": "test_external_linker_plugin",
    "linker_extra_config": {"nested": {"directory": "/tmp/kv"}},
}

_PLUGIN = """
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker

class ExternalLinker(UnifiedCacheLinker):
    def __init__(self, server_args, params, *, components, extra_config, forward_stream):
        self.args = (server_args, params, components, extra_config)
        self.layer_done_counter = object()
        self.forward_stream = forward_stream
        self.pools = {}

    def noop(self, *args):
        pass

    lookup = load = start_layer_wise_loading = cancel_queued_load = noop
    finish_layer_wise_loading = offload = noop
    num_completed_offloads = pop_completed_offload = reset = close = noop

class AbstractLinker(UnifiedCacheLinker):
    pass

not_a_class = 42
"""


def _create_startup_linker(*args, **kwargs):
    return create_unified_cache_linker(
        *args,
        backend=get_memory().unified_cache_external_linker_backend,
        config=get_startup_linker_config(),
        **kwargs,
    )


class TestExternalLinkerConfig(unittest.TestCase):
    def test_cli_json_and_programmatic_config(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--enable-unified-cache-external-linker",
                "--unified-cache-external-linker-backend",
                "dynamic",
                "--unified-cache-external-linker-config",
                json.dumps(_CONFIG),
            ]
        )
        self.assertEqual(parsed.unified_cache_external_linker_config, _CONFIG)
        self.assertEqual(parsed.unified_cache_external_linker_backend, "dynamic")
        args = ServerArgs(
            model_path="dummy",
            enable_unified_cache_external_linker=True,
            unified_cache_external_linker_backend="dynamic",
            unified_cache_external_linker_config=_CONFIG,
        )
        # Validation must not import the plugin in the frontend.
        with patch(
            "importlib.import_module", side_effect=AssertionError("plugin import")
        ):
            handle_hicache(args)

    def test_invalid_config(self):
        for value in (
            None,
            [],
            "{}",
            {},
            {**_CONFIG, "linker": " "},
            {**_CONFIG, "linker_module_path": 1},
            {**_CONFIG, "linker_extra_config": []},
            {**_CONFIG, "linker_extra_config": None},
            {**_CONFIG, "typo": True},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                UnifiedCacheLinkerConfig.from_dict(value)

    def test_disabled_linker_accepts_config_without_importing_plugin(self):
        for backend, config in (
            ("dynamic", _CONFIG),
            ("dynamic", None),
            ("mooncake", {"master_server_address": "localhost:50051"}),
            ("mori", {}),
        ):
            args = ServerArgs(
                model_path="dummy",
                enable_unified_cache_external_linker=False,
                unified_cache_external_linker_backend=backend,
                unified_cache_external_linker_config=config,
                disable_radix_cache=True,
            )
            with (
                self.subTest(backend=backend, config=config),
                patch(
                    "importlib.import_module",
                    side_effect=AssertionError("plugin import"),
                ),
            ):
                handle_hicache(args)

    def test_builtin_config_with_mooncake_mla_dedup(self):
        for backend in ("mooncake", "mori"):
            args = ServerArgs(
                model_path="dummy",
                enable_unified_cache_external_linker=True,
                unified_cache_external_linker_backend=backend,
                unified_cache_external_linker_config={"backend_option": "value"},
                enable_linker_mla_dedup=backend == "mooncake",
            )
            with (
                self.subTest(backend=backend),
                patch(
                    "importlib.import_module",
                    side_effect=AssertionError("plugin import"),
                ),
            ):
                handle_hicache(args)

    def test_invalid_config_in_server_args(self):
        for backend in ("dynamic", "mooncake", "mori"):
            for enabled in (False, True):
                for config in ([], "{}", 1):
                    with (
                        self.subTest(backend=backend, enabled=enabled, config=config),
                        self.assertRaisesRegex(ValueError, "must be a JSON object"),
                    ):
                        handle_hicache(
                            ServerArgs(
                                model_path="dummy",
                                enable_unified_cache_external_linker=enabled,
                                unified_cache_external_linker_backend=backend,
                                unified_cache_external_linker_config=config,
                            )
                        )
        for config in (None, {}, {"linker": "MissingModule"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                handle_hicache(
                    ServerArgs(
                        model_path="dummy",
                        enable_unified_cache_external_linker=True,
                        unified_cache_external_linker_backend="dynamic",
                        unified_cache_external_linker_config=config,
                    )
                )

    def test_conflicting_server_args(self):
        for overrides in (
            {"enable_hierarchical_cache": True},
            {"hicache_storage_backend": "file"},
            {"enable_linker_mla_dedup": True},
            {"disable_radix_cache": True},
            {"radix_cache_backend": "custom"},
        ):
            fields = dict(
                model_path="dummy",
                enable_unified_cache_external_linker=True,
                unified_cache_external_linker_backend="dynamic",
                unified_cache_external_linker_config=_CONFIG,
            )
            fields.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                handle_hicache(ServerArgs(**fields))


class TestExternalLinkerFactory(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        Path(directory.name, "test_external_linker_plugin.py").write_text(_PLUGIN)
        self.path_patch = patch.object(sys, "path", [directory.name, *sys.path])
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.addCleanup(sys.modules.pop, "test_external_linker_plugin", None)
        importlib.invalidate_caches()

    def _create(self, *, forward_stream=None, **config):
        with get_context().override_server_args(
            enable_unified_cache_external_linker=True,
            unified_cache_external_linker_backend="dynamic",
            unified_cache_external_linker_config={**_CONFIG, **config},
        ):
            return _create_startup_linker(
                "args", "params", components={"full"}, forward_stream=forward_stream
            )

    def test_passes_forward_stream_to_plugin(self):
        stream = object()
        self.assertIs(self._create(forward_stream=stream).forward_stream, stream)

    def test_loads_unregistered_module_and_copies_extra_config(self):
        linker = self._create()
        self.assertEqual(linker.args[:3], ("args", "params", {"full"}))
        self.assertEqual(linker.args[3], _CONFIG["linker_extra_config"])
        linker.args[3]["nested"]["directory"] = "changed"
        self.assertEqual(
            _CONFIG["linker_extra_config"]["nested"]["directory"], "/tmp/kv"
        )

    def test_invalid_class(self):
        for name in ("Missing", "not_a_class", "AbstractLinker"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "UnifiedCacheLinker"),
            ):
                self._create(linker=name)

    def test_missing_module(self):
        with self.assertRaises(ModuleNotFoundError):
            self._create(linker_module_path="no_such_external_linker_module")

    def test_plugin_dependency_error_is_preserved(self):
        with patch(
            "importlib.import_module",
            side_effect=ImportError("missing plugin dependency"),
        ):
            with self.assertRaisesRegex(ImportError, "missing plugin dependency"):
                self._create()

    def test_builtin_selection_and_private_config(self):
        stream = object()
        for backend, target in (
            (
                "mooncake",
                "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.MooncakeDirectLinker",
            ),
            (
                "mori",
                "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker.UMBPDirectLinker",
            ),
        ):
            # Even plugin-shaped config must not override the selected backend.
            for config in (None, {}, {"backend_option": {"value": 1}}, _CONFIG):
                with (
                    self.subTest(backend=backend, config=config),
                    get_context().override_server_args(
                        unified_cache_external_linker_config=config,
                        unified_cache_external_linker_backend=backend,
                    ),
                    patch(target) as constructor,
                ):
                    result = _create_startup_linker(
                        "args", "params", components={"full"}, forward_stream=stream
                    )
                    constructor.assert_called_once_with(
                        "args",
                        "params",
                        components={"full"},
                        extra_config=config or {},
                        forward_stream=stream,
                    )
                    self.assertIs(result, constructor.return_value)
                    passed_config = constructor.call_args.kwargs["extra_config"]
                    if config is not None:
                        self.assertIsNot(passed_config, config)
                    if config and "backend_option" in config:
                        passed_config["backend_option"]["value"] = 2
                        self.assertEqual(config["backend_option"]["value"], 1)

    def test_explicit_config_does_not_read_launch_defaults(self):
        with get_context().override_server_args(
            unified_cache_external_linker_backend="dynamic",
            unified_cache_external_linker_config=_CONFIG,
            hicache_storage_backend_extra_config='{"master_server_address": "old"}',
        ):
            for config in (None, {}):
                with (
                    self.subTest(config=config),
                    patch(
                        "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.MooncakeDirectLinker"
                    ) as constructor,
                ):
                    create_unified_cache_linker(
                        "args",
                        "params",
                        backend="mooncake",
                        config=config,
                        components={"full"},
                    )
                    self.assertEqual(constructor.call_args.kwargs["extra_config"], {})

    def test_dynamic_requires_config_in_worker(self):
        with (
            get_context().override_server_args(
                unified_cache_external_linker_backend="dynamic",
                unified_cache_external_linker_config=None,
            ),
            self.assertRaisesRegex(ValueError, "must be a JSON object"),
        ):
            _create_startup_linker("args", "params", components={"full"})


class TestBuiltinLinkerConfig(unittest.TestCase):
    def test_storage_receives_selected_config(self):
        pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[torch.zeros((2, 8), dtype=torch.uint8)]],
            layer_mapping={0: 0},
            page_size=1,
            rows_are_pages=False,
        )
        group = DevicePoolGroup([pool], num_layers=1, page_size=1)
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(),
            page_size=1,
        )
        legacy = {"master_server_address": "legacy:50051"}
        for backend, module_path in (
            ("mooncake", "mooncake_store.mooncake_direct_linker"),
            ("mori", "umbp.umbp_direct_linker"),
        ):
            module = importlib.import_module(
                f"sglang.srt.mem_cache.storage.{module_path}"
            )
            for config in (None, {}, {"master_server_address": "selected:50051"}):
                with (
                    self.subTest(backend=backend, config=config),
                    get_context().override_server_args(
                        unified_cache_external_linker_backend=backend,
                        unified_cache_external_linker_config=config,
                        hicache_storage_backend_extra_config=json.dumps(legacy),
                    ),
                    patch.object(
                        module, "resolve_hybrid_device_pool_group", return_value=group
                    ),
                    patch.object(
                        module.torch.distributed, "is_initialized", return_value=False
                    ),
                    # Stop before opening a store or starting device work.
                    patch.object(
                        module,
                        "HiCacheStorageConfig",
                        side_effect=RuntimeError("storage boundary"),
                    ) as storage_config,
                    self.assertRaisesRegex(RuntimeError, "storage boundary"),
                ):
                    _create_startup_linker("args", params, components={"full"})
                passed = storage_config.call_args.kwargs["extra_config"]
                expected = legacy if config is None else config
                self.assertEqual(
                    passed.get("master_server_address"),
                    expected.get("master_server_address"),
                )
                if backend == "mori":
                    self.assertIs(passed["ssd_enabled"], False)
                    self.assertNotIn("ssd_enabled", expected)
                if config == {}:
                    self.assertNotIn("master_server_address", passed)


if __name__ == "__main__":
    unittest.main()
