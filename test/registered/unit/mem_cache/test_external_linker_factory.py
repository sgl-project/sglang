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
from unittest.mock import patch

from sglang.srt.arg_groups.hicache_hook import handle_hicache
from sglang.srt.mem_cache.unified_cache.linker_config import UnifiedCacheLinkerConfig
from sglang.srt.mem_cache.unified_cache.linker_factory import (
    create_unified_cache_linker,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs

_CONFIG = {
    "linker": "ExternalLinker",
    "linker_module_path": "test_external_linker_plugin",
    "linker_extra_config": {"nested": {"directory": "/tmp/kv"}},
}

_PLUGIN = """
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker

class ExternalLinker(UnifiedCacheLinker):
    def __init__(self, server_args, params, *, components, extra_config):
        self.args = (server_args, params, components, extra_config)
        self.layer_done_counter = object()

    def noop(self, *args):
        pass

    lookup = load = start_layer_wise_loading = cancel_queued_load = noop
    num_completed_loads = pop_completed_load = offload = noop
    num_completed_offloads = pop_completed_offload = reset = close = noop

class AbstractLinker(UnifiedCacheLinker):
    pass

not_a_class = 42
"""


class TestExternalLinkerConfig(unittest.TestCase):
    def test_cli_json_and_programmatic_config(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--enable-unified-cache-external-linker",
                "--unified-cache-external-linker-config",
                json.dumps(_CONFIG),
            ]
        )
        self.assertEqual(parsed.unified_cache_external_linker_config, _CONFIG)
        args = ServerArgs(
            model_path="dummy",
            enable_unified_cache_external_linker=True,
            unified_cache_external_linker_config=_CONFIG,
        )
        # Validation must not import the plugin in the frontend.
        with patch(
            "importlib.import_module", side_effect=AssertionError("plugin import")
        ):
            handle_hicache(args)

    def test_invalid_config(self):
        for value in (
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

    def test_conflicting_server_args(self):
        for overrides in (
            {"enable_unified_cache_external_linker": False},
            {"enable_hierarchical_cache": True},
            {"hicache_storage_backend": "file"},
            {"enable_linker_mla_dedup": True},
            {"disable_radix_cache": True},
            {"radix_cache_backend": "custom"},
        ):
            fields = dict(
                model_path="dummy",
                enable_unified_cache_external_linker=True,
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

    def _create(self, **config):
        with get_context().override_server_args(
            enable_unified_cache_external_linker=True,
            unified_cache_external_linker_config={**_CONFIG, **config},
        ):
            return create_unified_cache_linker("args", "params", components={"full"})

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

    def test_builtins_keep_constructor_contract(self):
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
            with (
                self.subTest(backend=backend),
                get_context().override_server_args(
                    unified_cache_external_linker_config=None,
                    unified_cache_external_linker_backend=backend,
                ),
                patch(target) as constructor,
            ):
                result = create_unified_cache_linker(
                    "args", "params", components={"full"}
                )
                constructor.assert_called_once_with(
                    "args", "params", components={"full"}
                )
                self.assertIs(result, constructor.return_value)


if __name__ == "__main__":
    unittest.main()
