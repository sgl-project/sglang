import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSenderWrapper(CustomTestCase):
    @patch("sglang.srt.managers.scheduler_components.output_sender.sock_send")
    def test_propagates_route_from_scheduler_request(self, mock_sock_send):
        socket = MagicMock()
        scheduler_req = SimpleNamespace(http_worker_ipc="ipc:///tokenizer-worker-2")
        output = AbortReq(rid="overflow-request")

        SenderWrapper(socket).send_output(output, scheduler_req)

        self.assertEqual(output.http_worker_ipc, scheduler_req.http_worker_ipc)
        mock_sock_send.assert_called_once_with(socket, output)


if __name__ == "__main__":
    unittest.main()
