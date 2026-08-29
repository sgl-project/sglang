"""CPU unit test for benchmark server launch start method.

Regression guard for #34709: ``launch_or_reuse_server`` must start the child
server process with the *spawn* start method, not the platform-default ``fork``.

With ``fork``, the child inherits the accelerator context (CUDA/XPU) that the
parent CLI initialized while resolving ``ServerArgs`` and fails with
"Cannot re-initialize CUDA in forked subprocess" as soon as it touches the
device. Spawn gives the child a clean process.

Run: python3 test/srt/test_benchmark_endpoint.py
"""

import multiprocessing
import unittest
from unittest import mock

from sglang.benchmark.endpoint import launch_or_reuse_server


class _FakeServerArgs:
    """Minimal ServerArgs stand-in: only host/port are read before launch."""

    host = "127.0.0.1"
    port = 0

    def resolve_once(self):
        # The real resolution happens inside the spawned child; we never call it.
        pass


class LaunchUsesSpawnTest(unittest.TestCase):
    def test_launch_uses_spawn_not_fork(self):
        spawn_ctx = mock.MagicMock()
        spawn_proc = mock.MagicMock()
        spawn_proc.is_alive.return_value = True
        spawn_ctx.Process.return_value = spawn_proc

        with (
            mock.patch.object(
                multiprocessing,
                "get_context",
                side_effect=lambda method: spawn_ctx if method == "spawn" else None,
            ),
            mock.patch(
                "sglang.benchmark.endpoint.server_is_up", side_effect=[False, True]
            ),
            mock.patch("sglang.benchmark.endpoint.time.sleep"),
        ):
            launch_or_reuse_server(lambda server_args: None, _FakeServerArgs())

        # The child must be launched via the spawn context, not the default fork.
        spawn_ctx.Process.assert_called_once()
        _, kwargs = spawn_ctx.Process.call_args
        self.assertIsNotNone(kwargs.get("target"))


if __name__ == "__main__":
    unittest.main()
