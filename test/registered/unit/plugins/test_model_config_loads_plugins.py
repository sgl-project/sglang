"""
Unit tests for plugin loading ahead of model resolution.

``ServerArgs.__post_init__`` loads plugins, so an embedder that builds
``ServerArgs`` and resolves the model config before ``Engine.__init__`` still
gets its hooks applied. A hook on ``ModelConfig.__init__`` is only installed by
``HookRegistry.apply_hooks()``; if ``load_plugins()`` has not run by the time
the constructor is called, the hook is silently skipped.

Spawned processes receive ``ServerArgs`` by pickle, which bypasses
``__post_init__``, so their entry functions have to load plugins themselves.

Run:  python -m pytest test/registered/unit/plugins/test_model_config_loads_plugins.py -v
"""

from unittest.mock import MagicMock, patch

import sglang.srt.plugins as plugins_mod
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers import detokenizer_manager
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_TARGET = "sglang.srt.configs.model_config.ModelConfig.__init__"
_MODEL_PATH = "org/not-a-real-model"


class _HookFired(Exception):
    """Raised by the fake hook so ModelConfig.__init__ never runs."""


class TestServerArgsLoadsPlugins(CustomTestCase):
    def setUp(self):
        self._saved_loaded = plugins_mod._plugins_loaded
        self._saved_hooks = {k: list(v) for k, v in HookRegistry._hooks.items()}
        self._saved_patched = set(HookRegistry._patched)
        # HookRegistry.reset() clears bookkeeping but does not un-patch, so
        # the original constructor is restored by hand in tearDown.
        self._saved_init = ModelConfig.__dict__["__init__"]

        plugins_mod._plugins_loaded = False
        HookRegistry.reset()
        self.seen_model_paths = []

    def tearDown(self):
        setattr(ModelConfig, "__init__", self._saved_init)
        HookRegistry.reset()
        HookRegistry._hooks.update(self._saved_hooks)
        HookRegistry._patched.update(self._saved_patched)
        plugins_mod._plugins_loaded = self._saved_loaded

    def _fake_plugin(self):
        def before_init(*args, **kwargs):
            self.seen_model_paths.append(kwargs.get("model_path"))
            raise _HookFired()

        HookRegistry.register(_TARGET, before_init, HookType.BEFORE)

    def test_server_args_construction_loads_plugins(self):
        """Constructing ServerArgs loads plugins and applies hooks, so a
        BEFORE hook on ModelConfig.__init__ fires on the first model config
        build even though nobody called load_plugins()."""
        with patch.object(
            plugins_mod,
            "load_plugins_by_group",
            return_value={"fake": (self._fake_plugin, "fake-dist")},
        ) as mock_group:
            self.assertFalse(plugins_mod._plugins_loaded)
            self.assertNotIn(_TARGET, HookRegistry._patched)

            server_args = ServerArgs(model_path=_MODEL_PATH)

            self.assertTrue(plugins_mod._plugins_loaded)
            self.assertIn(_TARGET, HookRegistry._patched)
            self.assertEqual(mock_group.call_count, 1)
            self.assertEqual(self.seen_model_paths, [])

            with self.assertRaises(_HookFired):
                ModelConfig.from_server_args(server_args)
            self.assertEqual(self.seen_model_paths, [_MODEL_PATH])

            # A second record reuses the already-loaded plugins; discovery
            # is not repeated and the hook stays installed.
            ServerArgs(model_path=_MODEL_PATH)
            self.assertEqual(mock_group.call_count, 1)
            with self.assertRaises(_HookFired):
                ModelConfig.from_server_args(server_args)
            self.assertEqual(self.seen_model_paths, [_MODEL_PATH, _MODEL_PATH])


class TestDetokenizerProcessLoadsPlugins(CustomTestCase):
    def test_run_detokenizer_process_loads_plugins_before_manager(self):
        """run_detokenizer_process loads plugins before the manager (and its
        tokenizer) is constructed, without touching the parent process."""
        calls = []

        def fake_load_plugins():
            calls.append("load_plugins")

        def fake_manager_class(server_args, port_args):
            calls.append("manager")
            manager = MagicMock()
            manager.event_loop.side_effect = lambda: calls.append("event_loop")
            return manager

        serving = MagicMock()
        serving.tokenizer_worker_num = 1

        with (
            patch.object(detokenizer_manager, "load_plugins", fake_load_plugins),
            patch.object(detokenizer_manager, "kill_itself_when_parent_died"),
            patch.object(detokenizer_manager, "setproctitle"),
            patch.object(detokenizer_manager, "configure_logger"),
            patch.object(detokenizer_manager, "publish"),
            patch.object(detokenizer_manager, "psutil"),
            patch.object(detokenizer_manager, "get_serving", return_value=serving),
        ):
            detokenizer_manager.run_detokenizer_process(
                server_args=MagicMock(),
                port_args=MagicMock(),
                detokenizer_manager_class=fake_manager_class,
            )

        self.assertEqual(calls, ["load_plugins", "manager", "event_loop"])


if __name__ == "__main__":
    import unittest

    unittest.main()
