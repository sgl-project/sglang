"""Unit tests for request construction in the encode-disaggregation path."""

import asyncio
import threading
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.encoder.receiver import (
    MMReceiverBase,
    WaitingMMRequestStatus,
    WaitingRDMARequest,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEncodeReceiverRequestConstruction(CustomTestCase):
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
