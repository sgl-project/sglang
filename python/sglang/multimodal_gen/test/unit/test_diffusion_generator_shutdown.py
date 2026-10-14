import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.multimodal_gen.runtime.entrypoints import diffusion_generator as dg
from sglang.multimodal_gen.runtime.entrypoints.control_requests import ShutdownReq
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator


class _FakeProcess:
    name = "fake-worker"

    def __init__(self):
        self.alive = True
        self.join_timeouts = []
        self.terminated = False
        self.killed = False

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.alive = False


class TestDiffGeneratorShutdown(unittest.TestCase):
    def test_cli_shuts_down_on_success_and_generation_error(self):
        from sglang.multimodal_gen.runtime.entrypoints.cli import generate as cli

        for failure in (None, RuntimeError("generation failed")):
            with self.subTest(failure=failure):
                generator = object.__new__(DiffGenerator)
                generator.owns_scheduler_client = False
                generator.local_scheduler_process = None
                generator.generate = Mock(return_value=[], side_effect=failure)
                generator.shutdown = Mock()
                args = argparse.Namespace(config=None, diffusers_kwargs=None)
                server_args = SimpleNamespace(model_path="dummy/model")
                sampling_cls = SimpleNamespace(
                    get_cli_args=Mock(return_value={"prompt": "a white clock"})
                )
                with (
                    patch.object(
                        cli.ServerArgs, "from_cli_args", return_value=server_args
                    ),
                    patch.object(
                        cli,
                        "_resolve_cli_sampling_params_cls",
                        return_value=sampling_cls,
                    ),
                    patch.object(cli, "_apply_output_file_path_override"),
                    patch.object(
                        cli.DiffGenerator, "from_pretrained", return_value=generator
                    ),
                    patch.object(cli, "maybe_dump_performance") as dump,
                ):
                    if failure is None:
                        cli.generate_cmd(args)
                        dump.assert_called_once()
                    else:
                        with self.assertRaisesRegex(RuntimeError, "generation failed"):
                            cli.generate_cmd(args)
                        dump.assert_not_called()
                generator.shutdown.assert_called_once_with()

    def test_shutdown_uses_bounded_scheduler_timeout_and_forces_worker(self):
        generator = object.__new__(DiffGenerator)
        process = _FakeProcess()
        generator.local_scheduler_process = [process]
        generator.owns_scheduler_client = True

        client = SimpleNamespace(
            forward=Mock(side_effect=TimeoutError("blocked")),
            close=Mock(),
        )

        with patch.object(dg, "sync_scheduler_client", client):
            generator.shutdown()

        request = client.forward.call_args.args[0]
        self.assertIsInstance(request, ShutdownReq)
        self.assertEqual(client.forward.call_args.kwargs, {"timeout_ms": 5000})
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertAlmostEqual(process.join_timeouts[0], 10, delta=0.1)
        self.assertEqual(process.join_timeouts[1:], [1, 1])
        self.assertIsNone(generator.local_scheduler_process)
        self.assertFalse(generator.owns_scheduler_client)
        client.close.assert_called_once_with()

    def test_del_forces_cleanup_without_scheduler_request(self):
        generator = object.__new__(DiffGenerator)
        process = _FakeProcess()
        generator.local_scheduler_process = [process]
        generator.owns_scheduler_client = True

        client = SimpleNamespace(
            forward=Mock(),
            close=Mock(),
        )

        with patch.object(dg, "sync_scheduler_client", client):
            generator.__del__()

        client.forward.assert_not_called()
        client.close.assert_called_once_with()
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertIsNone(generator.local_scheduler_process)
        self.assertFalse(generator.owns_scheduler_client)

    def test_del_tolerates_missing_shutdown_globals(self):
        generator = object.__new__(DiffGenerator)
        process = _FakeProcess()
        generator.local_scheduler_process = [process]
        generator.owns_scheduler_client = True

        with (
            patch.object(dg, "logger", None),
            patch.object(dg, "sync_scheduler_client", None),
        ):
            generator.__del__()

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertIsNone(generator.local_scheduler_process)
        self.assertFalse(generator.owns_scheduler_client)


if __name__ == "__main__":
    unittest.main()
