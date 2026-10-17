"""
Unit tests for plugin loading ahead of model resolution.

``ServerArgs.__post_init__`` loads plugins, so an embedder that builds
``ServerArgs`` and resolves the model config before ``Engine.__init__`` still
gets its hooks applied. A hook on ``ModelConfig.__init__`` is only installed by
``HookRegistry.apply_hooks()``; if ``load_plugins()`` has not run by the time
the constructor is called, the hook is silently skipped.

Pickle and deepcopy rebuild through ServerArgs construction, so spawned workers
load their own plugins before model and tokenizer reads.

Run:  python -m pytest test/registered/unit/plugins/test_model_config_loads_plugins.py -v
"""

import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import sglang.srt.plugins as plugins_mod
from sglang.srt.configs.model_config import ModelConfig
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

    def test_unpickling_server_args_loads_plugins(self):
        with patch.object(plugins_mod, "load_plugins_by_group", return_value={}):
            payload = pickle.dumps(ServerArgs(model_path=_MODEL_PATH))

        with tempfile.TemporaryDirectory() as directory:
            plugin_dir = Path(directory)
            (plugin_dir / "fake_server_args_plugin.py").write_text(
                "import os\n"
                "from pathlib import Path\n\n"
                "def register():\n"
                "    Path(__file__).with_suffix('.loaded').write_text(str(os.getpid()))\n"
            )
            dist_info = plugin_dir / "fake_plugin-0.0.0.dist-info"
            dist_info.mkdir()
            (dist_info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: fake-plugin\nVersion: 0.0.0\n"
            )
            (dist_info / "entry_points.txt").write_text(
                "[sglang.srt.plugins]\n"
                "server_args_pickle_fixture = fake_server_args_plugin:register\n"
            )
            marker = plugin_dir / "fake_server_args_plugin.loaded"
            source_root = Path(__file__).resolve().parents[4] / "python"
            script = textwrap.dedent(
                """\
                import os
                import pickle
                import sys
                from pathlib import Path

                import sglang.srt.plugins as plugins_mod
                import sglang.srt.server_args as server_args_mod

                marker = Path(sys.argv[1])
                assert Path(server_args_mod.__file__).resolve() == Path(sys.argv[2])
                assert not plugins_mod._plugins_loaded
                assert not marker.exists(), "General plugin loaded before unpickling"
                server_args = pickle.loads(sys.stdin.buffer.read())
                assert marker.exists(), "General plugin was not loaded during ServerArgs unpickling"
                assert marker.read_text() == str(os.getpid())
                assert plugins_mod._plugins_loaded
                assert isinstance(server_args, server_args_mod.ServerArgs)
                assert server_args.model_path == sys.argv[3]
                """
            )
            env = os.environ.copy()
            env.pop("SGLANG_PLATFORM", None)
            env["SGLANG_PLUGINS"] = "server_args_pickle_fixture"
            env["PYTHONPATH"] = os.pathsep.join(
                (str(plugin_dir), str(source_root), env.get("PYTHONPATH", ""))
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(marker),
                    str(source_root / "sglang/srt/server_args.py"),
                    _MODEL_PATH,
                ],
                input=payload,
                capture_output=True,
                env=env,
                timeout=60,
            )
            self.assertEqual(
                result.returncode,
                0,
                result.stdout.decode() + result.stderr.decode(),
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
