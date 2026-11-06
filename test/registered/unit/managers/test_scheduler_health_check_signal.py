"""Health acknowledgements reach the frontend that owns the probe.

The Python frontend reads the acknowledgement from the zmq tokenizer socket.
The embedded Rust server does not consume that socket, so in Rust mode the
acknowledgement goes on its response ring instead.
"""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import HealthCheckOutput  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _ack(rid="HEALTH_CHECK_a", http_worker_ipc="ipc-a"):
    return HealthCheckOutput(
        rid=rid, http_worker_ipc=http_worker_ipc, incarnation="inc", completion_seq=1
    )


class TestHealthCheckSignal(CustomTestCase):
    def _scheduler(self, rust_server) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.rust_server = rust_server
        s.ipc_channels = MagicMock()
        return s

    def test_rust_mode_acknowledges_on_the_ring(self):
        server = MagicMock()
        s = self._scheduler(RustServer(server, http_port=0))

        s._send_health_ack(_ack())

        server.push_health_ack.assert_called_once_with()
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_python_mode_acknowledges_on_zmq(self):
        s = self._scheduler(None)
        ack = _ack()

        s._send_health_ack(ack)

        s.ipc_channels.send_to_tokenizer.send_output.assert_called_once_with(ack)


if __name__ == "__main__":
    unittest.main()
