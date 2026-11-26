"""Unit tests for the encode-disaggregation receiver."""

import threading
import time
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.encoder import receiver as receiver_module
from sglang.srt.disaggregation.encoder.receiver import (
    MMReceiverGrpc,
    MMReceiverHTTP,
    WaitingMMRequestStatus,
    _select_mm_processor_prompt,
)
from sglang.srt.managers.io_struct import AbortReq, TokenizedGenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.multimodal.modality import Modality
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


class TestGrpcThreadedEncodeDispatch(CustomTestCase):
    def test_threaded_dispatch_sends_grpc_encode(self):
        """The threaded dispatch used by zmq_to_scheduler reaches the gRPC encoder
        instead of reporting a dispatch error for every multimodal request."""
        receiver = MMReceiverGrpc.__new__(MMReceiverGrpc)
        receiver.encode_urls = ["grpc://127.0.0.1:1"]
        receiver.host = "127.0.0.1"
        sent = []
        dispatch_errors = []
        dispatch_ready = threading.Event()
        dispatch_ready.set()

        with patch.object(
            receiver_module,
            "_grpc_encode_request",
            side_effect=lambda target, encode_request: sent.append(
                (target, encode_request)
            ),
        ):
            receiver._run_encode_in_thread(
                req_id="rid-0",
                mm_data=[{"modality": Modality.IMAGE, "url": "http://host/a.png"}],
                endpoint_encode="encode",
                num_items_assigned={Modality.IMAGE: [1]},
                time_stats_json="{}",
                scheduler_dispatch_ready=dispatch_ready,
                on_dispatch_error=dispatch_errors.append,
            )

        self.assertEqual(dispatch_errors, [])
        self.assertEqual(len(sent), 1)
        target, encode_request = sent[0]
        self.assertEqual(target, "127.0.0.1:1")
        self.assertEqual(encode_request["mm_items"], ["http://host/a.png"])
        self.assertEqual(encode_request["req_id"], "rid-0")


class TestWaitingRequestTimeout(CustomTestCase):
    def test_a_result_reached_in_the_timeout_tick_is_kept(self):
        receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
        receiver.recv_timeout = 0.0
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

    def test_an_abort_in_the_completion_tick_stops_the_waiting_request(self):
        """An AbortReq arriving in the tick its waiting request receives its
        embeddings must abort the request instead of letting it trail the
        AbortReq in new_recv_reqs and enter the scheduler."""
        receiver = _abort_test_receiver()
        waiting_req = _completing_waiting_req(SimpleNamespace(rid="r"))
        receiver.waiting_list = [waiting_req]
        receiver.waiting_by_rid = {"r": waiting_req}
        abort = AbortReq(rid="r")

        with patch("torch.distributed.all_reduce"):
            new_recv_reqs, abort_reqs = receiver._process_waiting_requests(
                [abort], None
            )

        self.assertEqual(new_recv_reqs, [abort])
        self.assertEqual(abort_reqs, [(waiting_req.recv_req, "Aborted by user", 400)])
        self.assertEqual(receiver.waiting_list, [])
        self.assertEqual(receiver.waiting_by_rid, {})

    def test_an_abort_batched_after_its_request_stops_the_request(self):
        """A request and its AbortReq read in one batch: the request starts
        waiting, then the AbortReq after it must abort it before it receives its
        embeddings in the same tick and trails the AbortReq into the scheduler."""
        receiver = _abort_test_receiver()
        receiver.waiting_list = []
        receiver.waiting_by_rid = {}
        receiver.mm_processor = None
        receiver.model_type = None
        receiver.hostname = "localhost"
        receiver.scheduler_recv_socket = None
        receiver.scheduler_context = None
        receiver.scheduler_embedding_port = 0
        recv_req = Mock(
            spec=TokenizedGenerateReqInput,
            rid="r",
            need_wait_for_mm_inputs=True,
            encoder_urls=["http://encoder"],
        )
        started = []

        def waiting_cls(*, rid, recv_req, **kwargs):
            waiting_req = _completing_waiting_req(recv_req)
            waiting_req.send_encode_request = Mock()
            started.append(waiting_req)
            return waiting_req

        abort = AbortReq(rid="r")

        with patch("torch.distributed.all_reduce"):
            new_recv_reqs, abort_reqs = receiver._process_waiting_requests(
                [recv_req, abort], waiting_cls
            )

        self.assertEqual(len(started), 1)
        self.assertEqual(new_recv_reqs, [abort])
        self.assertEqual(abort_reqs, [(recv_req, "Aborted by user", 400)])
        self.assertEqual(receiver.waiting_list, [])
        self.assertEqual(receiver.waiting_by_rid, {})

    def test_an_abort_applies_only_to_requests_received_before_it(self):
        """With [abort_all, new_request] in one poll, the new request must keep
        waiting for its embeddings; the forwarded AbortReq reaching the
        scheduler must not abort it a second time."""
        receiver = _abort_test_receiver()
        receiver.waiting_list = []
        receiver.waiting_by_rid = {}
        receiver.mm_processor = None
        receiver.model_type = None
        receiver.hostname = "localhost"
        receiver.scheduler_recv_socket = None
        receiver.scheduler_context = None
        receiver.scheduler_embedding_port = 0
        recv_req = Mock(
            spec=TokenizedGenerateReqInput,
            rid="new",
            need_wait_for_mm_inputs=True,
            encoder_urls=["http://encoder"],
        )

        def waiting_cls(*, rid, recv_req, **kwargs):
            waiting_req = _completing_waiting_req(recv_req)
            waiting_req.start_time = time.time()
            waiting_req._try_recv_mm_data = lambda: None
            waiting_req.send_encode_request = Mock()
            return waiting_req

        abort = AbortReq(abort_all=True)

        with patch("torch.distributed.all_reduce"):
            new_recv_reqs, abort_reqs = receiver._process_waiting_requests(
                [abort, recv_req], waiting_cls
            )
        scheduler = SimpleNamespace(
            mm_receiver=receiver,
            request_lists=lambda: {},
            collect_inflight_reqs=lambda: [],
        )
        for req in new_recv_reqs:
            Scheduler.abort_request(scheduler, req)

        self.assertEqual(new_recv_reqs, [abort])
        self.assertEqual(abort_reqs, [])
        self.assertEqual(len(receiver.waiting_list), 1)
        self.assertEqual(
            receiver.waiting_list[0].status, WaitingMMRequestStatus.PENDING
        )


def _abort_test_receiver():
    receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
    receiver.recv_timeout = 10.0
    receiver.tp_size = 1
    receiver.tp_group = SimpleNamespace(cpu_group=None)
    receiver._drain_scheduler_embeddings = Mock()
    receiver.create_req = lambda recv_req: recv_req
    return receiver


def _completing_waiting_req(recv_req):
    """A waiting request that receives its embeddings on its first receive
    attempt unless it was failed first."""
    waiting_req = SimpleNamespace(
        rid=recv_req.rid,
        recv_req=recv_req,
        start_time=0.0,
        status=WaitingMMRequestStatus.PENDING,
        error_msg=None,
        error_code=None,
        release_resources=Mock(),
        close_recv_socket=Mock(),
    )

    def fail_and_release(error_msg, error_code=None):
        waiting_req.error_msg = error_msg
        waiting_req.error_code = error_code
        waiting_req.status = WaitingMMRequestStatus.FAIL
        waiting_req.release_resources()
        waiting_req.close_recv_socket()

    def recv():
        if waiting_req.status == WaitingMMRequestStatus.PENDING:
            waiting_req.status = WaitingMMRequestStatus.SUCCESS

    waiting_req._fail_and_release = fail_and_release
    waiting_req._try_recv_mm_data = recv
    return waiting_req


if __name__ == "__main__":
    unittest.main()
