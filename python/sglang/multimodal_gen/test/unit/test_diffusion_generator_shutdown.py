import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.multimodal_gen.runtime.entrypoints import diffusion_generator as dg
from sglang.multimodal_gen.runtime.entrypoints.cli import generate as cli_generate
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
    def test_generate_cmd_shuts_down_generator_on_success_and_error(self):
        server_args = SimpleNamespace(model_path="dummy", load_diffusion_decoder=False)
        sampling_params_cls = Mock()
        sampling_params_cls.get_cli_args.return_value = {"prompt": "test prompt"}
        cli_args = SimpleNamespace(
            config=None,
            diffusers_kwargs=None,
            output_file_path=None,
            perf_dump_path=None,
        )
        succeeded, failed = Mock(), Mock()
        failed.generate.side_effect = RuntimeError("generation failed")
        with (
            patch.object(
                cli_generate.ServerArgs, "from_cli_args", return_value=server_args
            ),
            patch.object(
                cli_generate,
                "_resolve_cli_sampling_params_cls",
                return_value=sampling_params_cls,
            ),
            patch.object(
                cli_generate.DiffGenerator,
                "from_pretrained",
                side_effect=[succeeded, failed],
            ),
        ):
            cli_generate.generate_cmd(cli_args)
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                cli_generate.generate_cmd(cli_args)
        succeeded.shutdown.assert_called_once_with()
        failed.shutdown.assert_called_once_with()

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
