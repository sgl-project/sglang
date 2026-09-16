"""Unit tests for sglang/benchmark/endpoint.py"""

import multiprocessing
import unittest
from unittest import mock

from sglang.benchmark.endpoint import launch_or_reuse_server
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _noop_launch_server(server_args):
    """Module-level, not a lambda: spawn pickles ``target`` at Process.start()."""


class _StubServerArgs:
    """resolve_once() is where a real record probes the device; stubbing it is what
    keeps this on CPU. Nothing but host/port is read before the child is created."""

    host = "127.0.0.1"
    port = 30000

    def resolve_once(self):
        pass


class TestLaunchOrReuseServer(CustomTestCase):
    def test_server_child_is_a_spawn_process(self):
        """#34709: the child must be spawned -- a fork inherits the accelerator
        context resolve_once() left in the parent and dies on its first device op.
        """
        with (
            # Patch BaseProcess, not get_context: mocking the context away would
            # erase the start method under test. Only the launch is suppressed.
            mock.patch.object(multiprocessing.process.BaseProcess, "start"),
            mock.patch.object(
                multiprocessing.process.BaseProcess, "is_alive", return_value=True
            ),
            # Down for the reuse probe, up on the first startup poll; extend the
            # list if the startup loop ever gains another probe.
            mock.patch(
                "sglang.benchmark.endpoint.server_is_up", side_effect=[False, True]
            ),
        ):
            proc, _ = launch_or_reuse_server(_noop_launch_server, _StubServerArgs())

        self.assertIsInstance(proc, multiprocessing.context.SpawnProcess)


if __name__ == "__main__":
    unittest.main()
