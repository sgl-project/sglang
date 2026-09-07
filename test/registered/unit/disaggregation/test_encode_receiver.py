"""Unit tests for the encode-disaggregation receiver."""

import asyncio
import threading
import time
import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.encoder.receiver import (
    MMReceiverBase,
    WaitingMMRequestStatus,
    WaitingRDMARequest,
    WaitingZmqRequest,
    WaitingZmqRequestGrpc,
    _ReceiveRegistrationRunner,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import EncoderDispatchErrorReq
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_registration_request(request_cls):
    request = request_cls.__new__(request_cls)
    request.rid = "registration-test"
    request.registration_runner = _ReceiveRegistrationRunner(
        "test-encoder-receive-registration"
    )
    request.registration_future = None
    request.registration_error = None
    request.registration_lock = threading.Lock()
    request.status = WaitingMMRequestStatus.PENDING
    request.error_msg = None
    request.error_code = None
    request.embedding_pool = None
    request.embeddings_buffer = None
    request.recv_embedding_data = None
    request._pool_slot_id = None
    request._mm_finalizer = None
    request.recv_socket = None
    request.recv_req = SimpleNamespace(rid=request.rid)
    request.num_items_assigned = {Modality.IMAGE: [1]}
    request.encoder_urls = ["http://encoder"]
    request.host_name = "127.0.0.1"
    request.receive_count = 1
    request.embedding_port = 12345
    return request


def _cancel_registration(request):
    future = request.registration_future
    request.release_resources()
    deadline = time.monotonic() + 1
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert future.cancelled()


class BlockingResponse:
    def __init__(self, started):
        self.started = started

    async def __aenter__(self):
        self.started.set()
        await asyncio.Event().wait()

    async def __aexit__(self, *args):
        return False


class BlockingSession:
    started = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return BlockingResponse(self.started)


class FailingResponse:
    async def __aenter__(self):
        raise ConnectionError("encoder unavailable")

    async def __aexit__(self, *args):
        return False


class FailingSession(BlockingSession):
    def post(self, *args, **kwargs):
        return FailingResponse()


class TestReceiveRegistration(CustomTestCase):
    def test_http_registration_does_not_block_scheduler(self):
        started = threading.Event()
        BlockingSession.started = started
        request = _make_registration_request(WaitingZmqRequest)
        with patch(
            "sglang.srt.disaggregation.encoder.receiver.aiohttp.ClientSession",
            BlockingSession,
        ):
            scheduler_call = threading.Thread(
                target=request.send_encode_request, daemon=True
            )
            scheduler_call.start()
            self.assertTrue(started.wait(timeout=1))
            scheduler_call.join(timeout=0.1)

        self.assertFalse(scheduler_call.is_alive())
        self.assertEqual(request.status, WaitingMMRequestStatus.PENDING)
        _cancel_registration(request)

    def test_grpc_registration_does_not_block_scheduler(self):
        started = threading.Event()

        async def blocking_registration(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        request = _make_registration_request(WaitingZmqRequestGrpc)
        with patch(
            "sglang.srt.disaggregation.encoder.receiver._grpc_scheduler_receive_url",
            blocking_registration,
        ):
            scheduler_call = threading.Thread(
                target=request.send_encode_request, daemon=True
            )
            scheduler_call.start()
            self.assertTrue(started.wait(timeout=1))
            scheduler_call.join(timeout=0.1)

        self.assertFalse(scheduler_call.is_alive())
        self.assertEqual(request.status, WaitingMMRequestStatus.PENDING)
        _cancel_registration(request)

    def test_failure_is_request_local(self):
        request = _make_registration_request(WaitingZmqRequest)
        with patch(
            "sglang.srt.disaggregation.encoder.receiver.aiohttp.ClientSession",
            FailingSession,
        ):
            request.send_encode_request()
            deadline = time.monotonic() + 1
            while request.status == WaitingMMRequestStatus.PENDING:
                self.assertLess(time.monotonic(), deadline)
                request._try_recv_mm_data()
                time.sleep(0.01)

        self.assertEqual(request.status, WaitingMMRequestStatus.FAIL)
        self.assertEqual(request.error_code, HTTPStatus.BAD_GATEWAY)
        self.assertIn("encoder unavailable", request.error_msg)


class TestEncodeReceiverRequestConstruction(CustomTestCase):
    def test_early_dispatch_error_waits_for_scheduler_request(self):
        encode_finished = threading.Event()
        scheduler_dispatch_ready = threading.Event()
        reported = []
        failure = EncoderDispatchErrorReq(
            rid="request-1",
            error_msg="encoder unavailable",
            error_code=HTTPStatus.BAD_GATEWAY,
        )

        async def fail_encode(**kwargs):
            encode_finished.set()
            return failure

        receiver = SimpleNamespace(encode=fail_encode)
        worker = threading.Thread(
            target=MMReceiverBase._run_encode_in_thread,
            args=(
                receiver,
                failure.rid,
                [],
                "encode",
                {},
                [],
                None,
                scheduler_dispatch_ready,
                reported.append,
            ),
        )
        worker.start()

        self.assertTrue(encode_finished.wait(timeout=1))
        worker.join(timeout=0.05)
        self.assertTrue(worker.is_alive())
        self.assertEqual(reported, [])

        scheduler_dispatch_ready.set()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(reported, [failure])

    def test_dispatch_error_fails_only_owning_wait(self):
        class WaitingRequest:
            def __init__(self, rid):
                self.rid = rid
                self.recv_req = SimpleNamespace(rid=rid)
                self.status = WaitingMMRequestStatus.PENDING
                self.error_msg = None
                self.error_code = None
                self.start_time = 0

            def _try_recv_mm_data(self):
                pass

            def _fail_and_release(self, error_msg, error_code=None):
                self.error_msg = error_msg
                self.error_code = error_code
                self.status = WaitingMMRequestStatus.FAIL

            def release_resources(self):
                pass

            def close_recv_socket(self):
                pass

        owner = WaitingRequest("request-1")
        other = WaitingRequest("request-2")
        receiver = SimpleNamespace(
            waiting_list=[owner, other],
            waiting_by_rid={owner.rid: owner, other.rid: other},
            scheduler_recv_socket=None,
            wait_timeout=float("inf"),
            tp_group=SimpleNamespace(cpu_group=object()),
            _drain_scheduler_embeddings=lambda: None,
            _sync_fail_info_across_tp=lambda request: None,
            create_req=lambda request: request,
        )
        dispatch_error = EncoderDispatchErrorReq(
            rid=owner.rid,
            error_msg="bad media",
            error_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

        with patch("torch.distributed.all_reduce"):
            _, abort_reqs = MMReceiverBase._process_waiting_requests(
                receiver, [dispatch_error], waiting_cls=None
            )

        self.assertEqual(owner.status, WaitingMMRequestStatus.FAIL)
        self.assertEqual(owner.error_msg, dispatch_error.error_msg)
        self.assertEqual(owner.error_code, dispatch_error.error_code)
        self.assertEqual(other.status, WaitingMMRequestStatus.PENDING)
        self.assertEqual([req.rid for req, _, _ in abort_reqs], [owner.rid])

    def test_extra_key_and_cache_salt_are_forwarded(self):
        scheduler = SimpleNamespace(
            model_config=SimpleNamespace(hf_eos_token_id={2}, vocab_size=128),
            disaggregation_mode=DisaggregationMode.NULL,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            metrics_collector=None,
            dllm_config=None,
            tokenizer=object(),
        )
        receiver = SimpleNamespace(scheduler=scheduler)
        recv_req = SimpleNamespace(
            rid="request-1",
            input_text="hello",
            input_ids=array("q", [1, 2]),
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            lora_id=None,
            input_embeds=None,
            custom_logit_processor=None,
            require_reasoning=False,
            return_hidden_states=False,
            return_routed_experts=False,
            routed_experts_start_len=0,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
            routed_dp_rank=None,
            disagg_prefill_dp_rank=None,
            priority=None,
            extra_key="classification",
            cache_salt="tenant-a",
            http_worker_ipc=None,
        )

        req = MMReceiverBase.create_req(receiver, recv_req)

        self.assertEqual(req.extra_key, "classification")
        self.assertEqual(req.cache_salt, "tenant-a")

    def test_rdma_worker_error_is_released_on_scheduler_thread(self):
        scheduler_thread = threading.get_ident()

        class ThreadCheckedSocket:
            closed_by = None

            def close(self):
                self.closed_by = threading.get_ident()

        recv_socket = ThreadCheckedSocket()
        request = WaitingRDMARequest.__new__(WaitingRDMARequest)
        request.rid = "request-1"
        request.status = WaitingMMRequestStatus.PENDING
        request.error_msg = None
        request.error_code = None
        request.recv_socket = recv_socket
        request._receive_error = None
        request._receive_error_lock = threading.Lock()
        request._buffer_lock = threading.Lock()
        request._terminal = False
        request._receive_running = False
        request.registration_future = None
        request.embeddings_buffer = None
        request._pool_slot_id = None
        request.embedding_pool = None
        request._mm_finalizer = None

        worker = threading.Thread(
            target=lambda: asyncio.run(
                request._check_encoder_responses(
                    [ConnectionError("encoder unavailable")], "/send"
                )
            )
        )
        worker.start()
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(request.status, WaitingMMRequestStatus.PENDING)
        self.assertIsNone(request.recv_socket.closed_by)

        request._try_recv_mm_data()

        self.assertEqual(request.status, WaitingMMRequestStatus.FAIL)
        self.assertIsNone(request.recv_socket)
        self.assertTrue(request._terminal)
        self.assertEqual(recv_socket.closed_by, scheduler_thread)

    def test_tp_peer_failure_closes_local_receive_socket(self):
        class WaitingRequest:
            rid = "request-1"
            recv_req = SimpleNamespace(rid=rid)
            status = WaitingMMRequestStatus.PENDING
            error_msg = "peer failed"
            error_code = None
            start_time = 0
            released = False
            closed = False

            def _try_recv_mm_data(self):
                pass

            def release_resources(self):
                self.released = True

            def close_recv_socket(self):
                self.closed = True

        waiting_req = WaitingRequest()
        receiver = SimpleNamespace(
            waiting_list=[waiting_req],
            waiting_by_rid={waiting_req.rid: waiting_req},
            scheduler_recv_socket=None,
            wait_timeout=float("inf"),
            tp_group=SimpleNamespace(cpu_group=object()),
            _drain_scheduler_embeddings=lambda: None,
            _sync_fail_info_across_tp=lambda request: None,
            create_req=lambda request: request,
        )

        def force_peer_failure(status, **kwargs):
            status.fill_(WaitingMMRequestStatus.FAIL)

        with patch("torch.distributed.all_reduce", force_peer_failure):
            _, abort_reqs = MMReceiverBase._process_waiting_requests(
                receiver, [], waiting_cls=None
            )

        self.assertTrue(waiting_req.released)
        self.assertTrue(waiting_req.closed)
        self.assertEqual(len(abort_reqs), 1)


if __name__ == "__main__":
    unittest.main()
