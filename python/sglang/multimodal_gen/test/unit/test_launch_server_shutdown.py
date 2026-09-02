import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.multimodal_gen.runtime import launch_server as ls
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints.utils import ShutdownReq


class _FakeProcess:
    name = "fake-worker"

    def __init__(self, *, exit_on_join: bool = False):
        self.alive = True
        self.exit_on_join = exit_on_join
        self.join_timeouts = []
        self.terminated = False
        self.killed = False

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)
        if self.exit_on_join:
            self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.alive = False


class TestLaunchServerShutdown(unittest.TestCase):
    def test_monolithic_shutdown_requests_scheduler_then_forces_worker(self):
        process = _FakeProcess()
        server_args = SimpleNamespace(disagg_role=RoleType.MONOLITHIC)
        client = Mock()

        with patch.object(ls, "SchedulerClient", return_value=client):
            ls.shutdown_scheduler_processes(server_args, [process])

        client.initialize.assert_called_once_with(server_args)
        request = client.forward.call_args.args[0]
        self.assertIsInstance(request, ShutdownReq)
        self.assertEqual(client.forward.call_args.kwargs, {"timeout_ms": 5000})
        client.close.assert_called_once_with()

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertAlmostEqual(process.join_timeouts[0], 10, delta=0.1)
        self.assertAlmostEqual(process.join_timeouts[1], 1, delta=0.1)
        self.assertAlmostEqual(process.join_timeouts[2], 1, delta=0.1)

    def test_scheduler_shutdown_error_still_forces_worker(self):
        process = _FakeProcess()
        server_args = SimpleNamespace(disagg_role=RoleType.MONOLITHIC)
        client = Mock()
        client.forward.side_effect = TimeoutError("blocked")

        with patch.object(ls, "SchedulerClient", return_value=client):
            ls.shutdown_scheduler_processes(server_args, [process])

        client.close.assert_called_once_with()
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_disagg_role_does_not_send_monolithic_shutdown_request(self):
        process = _FakeProcess(exit_on_join=True)
        server_args = SimpleNamespace(disagg_role=RoleType.ENCODER)

        with patch.object(ls, "SchedulerClient") as scheduler_client:
            ls.shutdown_scheduler_processes(server_args, [process])

        scheduler_client.assert_not_called()
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)


if __name__ == "__main__":
    unittest.main()
