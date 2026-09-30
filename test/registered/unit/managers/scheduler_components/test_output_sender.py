import asyncio
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_scheduler_req(http_worker_ipc: str) -> Req:
    return Req(
        rid="scheduler-request",
        origin_input_text="prompt",
        origin_input_ids=[1],
        sampling_params=SamplingParams(),
        http_worker_ipc=http_worker_ipc,
    )


class TestSenderWrapper(CustomTestCase):
    @patch("sglang.srt.managers.scheduler_components.output_sender.sock_send")
    def test_preserves_existing_output_route(self, mock_sock_send):
        socket = MagicMock()
        scheduler_req = _make_scheduler_req("ipc:///source-worker")
        output = AbortReq(
            rid="already-routed",
            http_worker_ipc="ipc:///destination-worker",
        )

        SenderWrapper(socket).send_output(output, scheduler_req)

        self.assertEqual(output.http_worker_ipc, "ipc:///destination-worker")
        mock_sock_send.assert_called_once_with(socket, output)

    @patch("sglang.srt.managers.scheduler_components.output_sender.sock_send")
    def test_scheduler_abort_routes_to_origin_worker(self, mock_sock_send):
        socket = MagicMock()
        worker_ipc = "ipc:///tokenizer-worker-2"
        output = AbortReq(rid="overflow-request")

        SenderWrapper(socket).send_output(output, _make_scheduler_req(worker_ipc))

        routed_output = mock_sock_send.call_args.args[1]
        router = MultiTokenizerRouter.__new__(MultiTokenizerRouter)
        router.socket_mapping = MagicMock()
        asyncio.run(router._distribute_result_to_workers(routed_output))

        router.socket_mapping.send_output.assert_called_once_with(
            worker_ipc, routed_output
        )


if __name__ == "__main__":
    unittest.main()
