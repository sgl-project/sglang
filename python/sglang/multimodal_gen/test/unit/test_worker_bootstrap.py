# SPDX-License-Identifier: Apache-2.0

import importlib
import multiprocessing as mp
import pathlib
import shutil
import sys
import tempfile
import unittest

from sglang.multimodal_gen.runtime import worker_bootstrap

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


# Real module rather than an embedded source string, so it is formatted, linted
# and parsed like any other file. It stays a top-level module with no diffusion
# imports of its own, which is what keeps the import-order measurement honest.
FAKE_PLUGIN_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "sgl_fake_plugin.py"


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
    from sglang.multimodal_gen import plugins

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
                master_port=0,
                server_args=worker_bootstrap.ServerArgsPayload.capture(server_args),
                pipe_writer=writer,
                task_pipe_r=None,
                result_pipe_w=None,
                task_pipes_to_slaves=[],
                result_pipes_from_slaves=[],
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


if __name__ == "__main__":
    unittest.main()
