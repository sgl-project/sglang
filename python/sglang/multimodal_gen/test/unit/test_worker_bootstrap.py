# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.managers import worker_bootstrap

WORKER_MODULE = "sglang.multimodal_gen.runtime.managers.gpu_worker"
GENERATOR_MODULE = "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator"
SERVER_ARGS_MODULE = "sglang.multimodal_gen.runtime.server_args.server_args"
HTTP_SERVER_MODULE = "sglang.multimodal_gen.runtime.launch_server"

METADATA = "Metadata-Version: 2.1\nName: sgl-fake-plugin\nVersion: 0.1\n"
ENTRY_POINTS = """\
[sglang.multimodal_gen.platforms]
fake = sgl_fake_plugin:activate
[sglang.multimodal_gen.plugins]
fake = sgl_fake_plugin:register
"""


# Real modules, not embedded source strings, so they are linted like any other
# file. The plugin imports no diffusion module of its own, which is what keeps
# the import-order measurement honest.
FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
FAKE_PLUGIN_FIXTURE = FIXTURES_DIR / "sgl_fake_plugin.py"
FACADE_IMPORT_SCRIPT = FIXTURES_DIR / "offline_script_facade_import.py"
RUNTIME_IMPORT_SCRIPT = FIXTURES_DIR / "offline_script_runtime_import.py"

PYTHON_ROOT = pathlib.Path(__file__).parents[4]
EARLY_IMPORT_WARNING = "imported before this worker initialized its platform"
SCRIPT_TIMEOUT_S = 300


def _install_fake_plugin_dist(root: pathlib.Path) -> None:
    shutil.copy(FAKE_PLUGIN_FIXTURE, root / "sgl_fake_plugin.py")
    dist_info = root / "sgl_fake_plugin-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(METADATA)
    (dist_info / "entry_points.txt").write_text(ENTRY_POINTS)


def _check_cli_import_order(pipe_writer) -> None:
    from sglang.multimodal_gen.runtime.entrypoints.cli import main as cli_main

    imported_before_activation = GENERATOR_MODULE in sys.modules

    class StopAtPluginBoundary(Exception):
        pass

    def stop_before_command_imports():
        raise StopAtPluginBoundary

    cli_main.apply_plugin_hooks = stop_before_command_imports
    try:
        cli_main.generate_cmd_init()
    except StopAtPluginBoundary:
        pass

    pipe_writer.send(
        {
            "imported_before_activation": imported_before_activation,
            "imported_after_failed_activation": GENERATOR_MODULE in sys.modules,
        }
    )
    pipe_writer.close()


def _check_http_server_import_order(pipe_writer) -> None:
    from sglang.multimodal_gen.runtime.platforms import plugins

    class StopAtPluginBoundary(Exception):
        pass

    observed = {
        "http_server_imported_before_bootstrap": HTTP_SERVER_MODULE in sys.modules,
        "server_args_imported_before_bootstrap": SERVER_ARGS_MODULE in sys.modules,
    }

    def stop_before_runtime_imports():
        observed.update(
            http_server_imported_when_hooks_applied=(HTTP_SERVER_MODULE in sys.modules),
            server_args_imported_when_hooks_applied=(SERVER_ARGS_MODULE in sys.modules),
        )
        raise StopAtPluginBoundary

    plugins.load_plugins = lambda: None
    plugins.apply_plugin_hooks = stop_before_runtime_imports
    try:
        worker_bootstrap.bootstrap_http_server_process(None)
    except StopAtPluginBoundary:
        pass

    pipe_writer.send(observed)
    pipe_writer.close()


class TestBootstrapImportBoundary(unittest.TestCase):
    def test_manager_namespace_does_not_hide_early_worker_imports(self):
        for module, warned in (
            ("sglang.multimodal_gen.runtime.managers", False),
            ("sglang.multimodal_gen.runtime.platforms.plugins", False),
            (worker_bootstrap.__name__, False),
            (WORKER_MODULE, True),
        ):
            modules = {
                "__main__": SimpleNamespace(__file__="offline.py"),
                module: None,
            }
            with (
                self.subTest(module=module),
                patch.object(worker_bootstrap, "sys", SimpleNamespace(modules=modules)),
                patch.object(worker_bootstrap.logging, "getLogger") as get_logger,
            ):
                worker_bootstrap._warn_if_runtime_imported_early()
                if warned:
                    warning = get_logger.return_value.warning
                    warning.assert_called_once()
                    self.assertEqual(warning.call_args.args[1], module)
                else:
                    get_logger.assert_not_called()


class TestSpawnedWorkerReceivesPluginOverride(unittest.TestCase):
    """End-to-end over a real spawn, with a real entry-point distribution.

    A spawned child re-imports from a blank interpreter, so nothing the parent
    patched survives. The child must initialize its backend, register its own
    hooks, and apply them before invoking the worker.
    """

    def test_lifecycle_precedes_worker_import_and_argument_materialization(self):
        # The parent has a real ServerArgs object; the process boundary must
        # keep its class opaque until bootstrap chooses to materialize it.
        # Imported before the fake dist reaches sys.path: this resolves the
        # platform, which must not land on a class living in a temp directory.
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _install_fake_plugin_dist(root)

            # Spawn ships sys.path to the child, so the dist is discoverable there.
            sys.path.insert(0, str(root))
            self.addCleanup(sys.path.remove, str(root))
            importlib.invalidate_caches()

            server_args = ServerArgs.__new__(ServerArgs)
            reader, writer = mp.Pipe(duplex=False)
            spec = worker_bootstrap.SchedulerProcessSpec(
                local_rank=0,
                rank=0,
                server_args=worker_bootstrap.ServerArgsPayload.capture(server_args),
                pipe_writer=writer,
            )

            process = mp.get_context("spawn").Process(
                target=worker_bootstrap.bootstrap_scheduler_process,
                args=(spec,),
            )
            process.start()
            writer.close()
            self.addCleanup(process.join, 10)
            self.addCleanup(process.kill)

            result = None
            if reader.poll(120):
                try:
                    result = reader.recv()
                except EOFError:
                    pass
            if result is None:
                process.join(10)
                self.fail(
                    "child sent nothing back, so the override never ran "
                    f"(exit code {process.exitcode})"
                )

        self.assertTrue(result["override_ran"], "plugin override did not run")
        self.assertIs(
            result["worker_imported_when_plugin_ran"],
            False,
            "plugins loaded after the worker module was already imported",
        )
        self.assertIs(
            result["generator_imported_when_plugin_ran"],
            False,
            "the package facade imported the diffusion runtime before plugins loaded",
        )
        self.assertIs(
            result["worker_imported_when_backend_initialized"],
            False,
            "worker imports preceded platform backend initialization",
        )
        self.assertIs(
            result["server_args_imported_when_backend_initialized"],
            False,
            "spawn unpickled ServerArgs before platform backend initialization",
        )
        self.assertIs(
            result["server_args_imported_when_plugin_ran"],
            False,
            "spawn materialized ServerArgs before plugin registration",
        )
        self.assertTrue(
            result["backend_initialized_when_plugin_ran"],
            "plugin registration ran before platform backend initialization",
        )
        self.assertTrue(
            result["backend_initialized"],
            "platform backend initialized after the worker override ran",
        )

    def test_cli_activates_plugins_before_importing_commands(self):
        reader, writer = mp.Pipe(duplex=False)
        process = mp.get_context("spawn").Process(
            target=_check_cli_import_order,
            args=(writer,),
        )
        process.start()
        writer.close()
        self.addCleanup(process.join, 10)
        self.addCleanup(process.kill)

        self.assertTrue(reader.poll(30), "child did not report CLI import state")
        result = reader.recv()
        self.assertFalse(result["imported_before_activation"])
        self.assertFalse(result["imported_after_failed_activation"])

    def test_http_hooks_apply_before_runtime_imports(self):
        reader, writer = mp.Pipe(duplex=False)
        process = mp.get_context("spawn").Process(
            target=_check_http_server_import_order,
            args=(writer,),
        )
        process.start()
        writer.close()
        self.addCleanup(process.join, 10)
        self.addCleanup(process.kill)

        self.assertTrue(reader.poll(30), "child did not report HTTP import state")
        result = reader.recv()
        self.assertEqual(
            result,
            {
                "http_server_imported_before_bootstrap": False,
                "server_args_imported_before_bootstrap": False,
                "http_server_imported_when_hooks_applied": False,
                "server_args_imported_when_hooks_applied": False,
            },
        )


class TestOfflineScriptImportContract(unittest.TestCase):
    """Real scripts in a real interpreter, because spawn re-executes the
    launching script's module scope before it unpickles anything."""

    def _run_offline_script(self, script: pathlib.Path):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            dist_root = root / "site"
            dist_root.mkdir()
            _install_fake_plugin_dist(dist_root)
            result_path = root / "observed.json"

            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                path
                for path in (
                    str(dist_root),
                    str(FIXTURES_DIR),
                    str(PYTHON_ROOT),
                    env.get("PYTHONPATH", ""),
                )
                if path
            )
            completed = subprocess.run(
                [sys.executable, str(script), str(result_path)],
                env=env,
                capture_output=True,
                text=True,
                timeout=SCRIPT_TIMEOUT_S,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(result_path.read_text())

        self.assertIsNotNone(
            payload["observed"],
            f"child sent nothing back (exit code {payload['exitcode']})",
        )
        return payload["observed"], completed.stderr

    def test_a_module_scope_facade_import_leaves_the_child_lifecycle_intact(self):
        observed, stderr = self._run_offline_script(FACADE_IMPORT_SCRIPT)

        self.assertTrue(observed["override_ran"], "plugin override did not run")
        self.assertIs(
            observed["generator_imported_when_plugin_ran"],
            False,
            "re-executing the script imported the generator before plugins loaded",
        )
        self.assertNotIn(EARLY_IMPORT_WARNING, stderr)

    def test_a_module_scope_runtime_import_is_reported_by_the_child(self):
        observed, stderr = self._run_offline_script(RUNTIME_IMPORT_SCRIPT)

        self.assertTrue(observed["override_ran"], "plugin override did not run")
        self.assertIs(
            observed["generator_imported_when_plugin_ran"],
            True,
            "the script layout under test no longer imports the runtime early",
        )
        self.assertIn(
            EARLY_IMPORT_WARNING,
            stderr,
            "the child accepted a mis-ordered import without reporting it",
        )
        self.assertIn(
            RUNTIME_IMPORT_SCRIPT.name,
            stderr,
            "the report did not name the script whose imports have to move",
        )


if __name__ == "__main__":
    unittest.main()
