"""A busy scheduler acknowledges the health probes it skipped after each step.

The Python frontend reads the acknowledgement from the zmq tokenizer socket.
The embedded Rust server does not consume that socket, so in Rust mode the
acknowledgement goes on its response ring instead.
"""

import unittest
from collections import deque
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import HealthCheckOutput  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHealthCheckSignal(CustomTestCase):
    def _scheduler(self, rust_server, pending) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.rust_server = rust_server
        s.ipc_channels = MagicMock()
        s.return_health_check_ipcs = deque(pending)
        return s

    def test_rust_mode_acknowledges_all_skipped_probes_on_the_ring(self):
        server = MagicMock()
        s = self._scheduler(RustServer(server, http_port=0), [None, None])

        s.maybe_send_health_check_signal()
        s.maybe_send_health_check_signal()

        server.push_health_ack.assert_called_once_with()
        self.assertFalse(s.return_health_check_ipcs)
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_python_mode_acknowledges_one_probe_per_step_on_zmq(self):
        s = self._scheduler(None, ["ipc-a", "ipc-b"])

        s.maybe_send_health_check_signal()

        (output,), _ = s.ipc_channels.send_to_tokenizer.send_output.call_args
        self.assertIsInstance(output, HealthCheckOutput)
        self.assertEqual(output.http_worker_ipc, "ipc-a")
        self.assertEqual(list(s.return_health_check_ipcs), ["ipc-b"])


if __name__ == "__main__":
    unittest.main()
