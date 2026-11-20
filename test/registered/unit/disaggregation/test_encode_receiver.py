"""Unit tests for the encode-disaggregation receiver."""

import time
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import msgspec
import numpy as np

from sglang.srt.disaggregation.encoder import receiver as receiver_module
from sglang.srt.disaggregation.encoder.receiver import (
    MMReceiverHTTP,
    WaitingMMRequestStatus,
    _select_mm_processor_prompt,
)
from sglang.srt.managers.io_struct import EncoderDispatchErrorReq
from sglang.srt.multimodal.modality import Modality
from sglang.srt.rust_server import server as server_module
from sglang.srt.rust_server.server import RustServer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestSelectMMProcessorPrompt(CustomTestCase):
    def test_tokens_in_prompt_is_list(self):
        """A tokens-in prompt reaches the processor as list[int], not array("q");
        HTTP sends input_text=None and the gRPC servicer sends ""."""
        processor = SimpleNamespace(prefer_tokenized_input=False)
        for input_text in (None, ""):
            with self.subTest(input_text=input_text):
                recv_req = SimpleNamespace(
                    input_text=input_text, input_ids=array("q", [1, 2, 3])
                )

                prompt = _select_mm_processor_prompt(recv_req, processor)

                self.assertIsInstance(prompt, list)
                self.assertEqual(prompt, [1, 2, 3])


class TestWaitingRequestTimeout(CustomTestCase):
    def test_a_result_reached_in_the_timeout_tick_is_kept(self):
        receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
        receiver.wait_timeout = 0.0
        receiver.waiting_by_rid = {}
        receiver.tp_group = SimpleNamespace(cpu_group=None)
        receiver._drain_scheduler_embeddings = Mock()
        receiver.create_req = Mock()
        waiting_req = SimpleNamespace(
            rid="r",
            recv_req=SimpleNamespace(rid="r"),
            start_time=0.0,
            status=WaitingMMRequestStatus.PENDING,
            release_resources=Mock(),
            close_recv_socket=Mock(),
        )

        def recv():
            waiting_req.status = WaitingMMRequestStatus.SUCCESS

        waiting_req._try_recv_mm_data = recv
        receiver.waiting_list = [waiting_req]

        with patch("torch.distributed.all_reduce"):
            new_recv_reqs, abort_reqs = receiver._process_waiting_requests([], None)

        self.assertEqual(new_recv_reqs, [waiting_req.recv_req])
        self.assertEqual(abort_reqs, [])
        waiting_req.release_resources.assert_not_called()


class TestRustServerEncoderDispatch(CustomTestCase):
    """A ``--language-only`` Rust server dispatches each drained image request
    through the same receiver code as Python's ``TokenizerManager``."""

    MEDIA = ["http://a/x.png", {"url": "data:image/png;base64,eA==", "detail": "high"}]

    def _server(self, encode_urls, dispatch_error=None):
        dispatcher = MMReceiverHTTP.__new__(MMReceiverHTTP)
        dispatcher.encode_urls = encode_urls
        dispatcher.encode = AsyncMock(return_value=dispatch_error)
        rust = RustServer(Mock(), http_port=0)
        rust._scheduler_loop_started.set()
        rust._encoder_dispatcher = dispatcher
        return rust

    @staticmethod
    def _drain(rust, requests):
        rust.server.recv_requests.return_value = [
            SimpleNamespace(header=obj.rid, buffers=buffers)
            for obj, buffers in requests
        ]
        decoded = {obj.rid: obj for obj, _ in requests}
        with patch(
            "sglang.srt.rust_server.server.msgpack_decode_explained",
            side_effect=decoded.__getitem__,
        ):
            return rust.drain(0)

    def _request(self, rid):
        obj = SimpleNamespace(rid=rid, input_ids=None, token_ids_logprob=None)
        buffers = [
            ("input_ids", np.array([1, 2, 3], dtype=np.int64)),
            (
                "epd.media",
                np.frombuffer(msgspec.msgpack.encode(self.MEDIA), dtype=np.uint8),
            ),
        ]
        return obj, buffers

    def test_drained_request_is_dispatched_and_its_error_follows_it(self):
        error = EncoderDispatchErrorReq(rid="r1", error_msg="boom", error_code=502)
        rust = self._server(["http://e0"], dispatch_error=error)
        obj, buffers = self._request("r1")

        self.assertEqual(self._drain(rust, [(obj, buffers)]), [obj])
        self.assertIs(obj.need_wait_for_mm_inputs, True)
        self.assertEqual(obj.num_items_assigned, {Modality.IMAGE: [2]})
        self.assertEqual(obj.encoder_urls, ["http://e0"])

        deadline = time.monotonic() + 10
        while not rust._encoder_dispatch_errors and time.monotonic() < deadline:
            time.sleep(0.01)
        rust.server.wake_request_wait.assert_called_once_with()
        rust.server.recv_requests.return_value = []
        self.assertEqual(rust.drain(0), [error])
        self.assertEqual(rust.drain(0), [])
        rust._encoder_dispatcher.encode.assert_awaited_once()
        self.assertEqual(
            rust._encoder_dispatcher.encode.await_args.kwargs["mm_data"],
            [
                {
                    "url": "http://a/x.png",
                    "modality": Modality.IMAGE,
                    "content_hash": None,
                },
                {
                    "url": "data:image/png;base64,eA==",
                    "modality": Modality.IMAGE,
                    "detail": "high",
                    "content_hash": None,
                },
            ],
        )

    def test_request_without_an_encoder_is_a_503(self):
        rust = self._server([])
        obj, buffers = self._request("r2")

        self.assertEqual(self._drain(rust, [(obj, buffers)]), [])
        rust.server.push_error.assert_not_called()
        header, payloads = rust.server.push_decode_result_batch.call_args.args
        self.assertEqual(payloads, [b""])
        columns = msgspec.msgpack.decode(header)
        self.assertEqual(columns[0], ["r2"])
        self.assertEqual(
            columns[1],
            [
                {
                    "type": "abort",
                    "message": "no encoder is registered with this "
                    "--language-only server. Pass --encoder-urls at launch, or "
                    "register an encoder with POST /register_encoder_url on the "
                    "encoder bootstrap port.",
                    "status_code": 503,
                }
            ],
        )
        self.assertEqual(columns[2], [3])
        rust._encoder_dispatcher.encode.assert_not_called()

    def test_bootstrap_server_and_dispatcher_share_one_url_list(self):
        rust = RustServer(Mock(), http_port=0)
        rust.server_args = None
        disagg = SimpleNamespace(encoder_urls=["http://e0"], encoder_bootstrap_port=1)
        with (
            patch.object(server_module, "get_disagg", return_value=disagg),
            patch.object(server_module, "get_serving"),
            patch.object(receiver_module, "EncoderBootstrapServer") as bootstrap,
            patch.object(receiver_module, "create_mm_receiver") as create,
        ):
            rust._start_encoder_dispatch()
        urls = bootstrap.call_args.kwargs["urls"]
        self.assertEqual(urls, ["http://e0"])
        self.assertIs(create.call_args.kwargs["encode_urls"], urls)


if __name__ == "__main__":
    unittest.main()
