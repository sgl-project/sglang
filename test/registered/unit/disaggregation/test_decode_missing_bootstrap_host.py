"""A PD decode request without `bootstrap_host` must not take the server down.

Decode builds the prefill's bootstrap address from `bootstrap_host` and
`bootstrap_port`, and `NetworkAddress(None, port)` raises. The request reaches
every TP rank, so an uncaught raise there kills every scheduler at once.

Under `--disaggregation-transfer-backend fake` a request without
`bootstrap_host` is valid: `_is_fake_transfer` accepts it and the fake receiver
takes no address. On a real backend decode cannot serve it, so intake rejects
that one request with a 400, as it already does a missing `bootstrap_room`.
"""

import unittest
from http import HTTPStatus
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.disaggregation.utils import KVClassType, TransferBackend
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import ListKind
from sglang.test.scheduler_test_utils import FakeReq, decode_queue

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PREFILL_HOST = "192.0.2.1"
BOOTSTRAP_PORT = 8998


def _server_backend(name: str):
    """The server's `--disaggregation-transfer-backend`, as `_is_fake_transfer`
    reads it."""
    return patch(
        "sglang.srt.disaggregation.utils.get_disagg",
        return_value=SimpleNamespace(disaggregation_transfer_backend=name),
    )


def _req(bootstrap_host):
    return FakeReq(
        rid="r",
        bootstrap_host=bootstrap_host,
        bootstrap_port=BOOTSTRAP_PORT,
        bootstrap_room=7,
        to_finish=None,
        finished_reason=None,
    )


def _prealloc_queue(transfer_backend: TransferBackend):
    q = SimpleNamespace(
        transfer_backend=transfer_backend,
        kv_manager=MagicMock(),
        queue=decode_queue(ListKind.PD_PREALLOC),
        pending_reqs=[],
        _check_if_req_exceed_kv_capacity=MagicMock(return_value=False),
    )
    q._create_receiver_and_enqueue = MethodType(
        DecodePreallocQueue._create_receiver_and_enqueue, q
    )
    return q


class TestDecodeReceiverBootstrapAddr(CustomTestCase):
    def test_fake_backend_admits_request_without_bootstrap_host(self):
        """Regression: the receiver was given a bootstrap address even for a
        fake transfer, and building it from `bootstrap_host=None` raised."""
        q = _prealloc_queue(TransferBackend.FAKE)
        req = _req(bootstrap_host=None)
        with _server_backend("fake"):
            DecodePreallocQueue.add(q, req)

        (decode_req,) = q.queue.items()
        self.assertIs(decode_req.req, req)
        self.assertIsInstance(decode_req.kv_receiver, FakeKVReceiver)
        # A fake transfer resolves no prefill DP rank and is ready at once.
        self.assertEqual(q.pending_reqs, [])
        self.assertEqual(decode_req.kv_receiver.poll(), KVPoll.WaitingForInput)

    def test_real_backend_receiver_gets_prefill_address(self):
        receiver_class = MagicMock()
        q = _prealloc_queue(TransferBackend.MOONCAKE)
        with (
            _server_backend("mooncake"),
            patch(
                "sglang.srt.disaggregation.decode.get_kv_class",
                return_value=receiver_class,
            ) as get_kv_class,
        ):
            DecodePreallocQueue._create_receiver_and_enqueue(q, _req(PREFILL_HOST))

        get_kv_class.assert_called_once_with(
            TransferBackend.MOONCAKE, KVClassType.RECEIVER
        )
        self.assertEqual(
            receiver_class.call_args.kwargs["bootstrap_addr"],
            f"{PREFILL_HOST}:{BOOTSTRAP_PORT}",
        )

    def test_fake_bootstrap_host_on_real_backend_uses_fake_receiver(self):
        """Warm-up and health-check requests set `FAKE_BOOTSTRAP_HOST`."""
        q = _prealloc_queue(TransferBackend.MOONCAKE)
        with _server_backend("mooncake"):
            decode_req = DecodePreallocQueue._create_receiver_and_enqueue(
                q, _req(FAKE_BOOTSTRAP_HOST)
            )
        self.assertIsInstance(decode_req.kv_receiver, FakeKVReceiver)


class _PastIntake(Exception):
    """Raised by the first scheduler step after intake validation."""


def _scheduler(mode: DisaggregationMode, transfer_backend: TransferBackend):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_session_radix_cache = False
    scheduler.model_config = SimpleNamespace(hf_eos_token_id={1}, vocab_size=128)
    scheduler.disaggregation_mode = mode
    scheduler.transfer_backend = transfer_backend
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler.output_streamer = MagicMock()
    scheduler._add_request_to_queue = MagicMock()
    scheduler._maybe_namespace_elastic_radix_cache = MagicMock(side_effect=_PastIntake)
    return scheduler


def _recv_req(bootstrap_host):
    return MagicMock(
        session_params=None,
        session_id=None,
        input_embeds=None,
        mm_inputs=None,
        bootstrap_host=bootstrap_host,
        bootstrap_port=BOOTSTRAP_PORT,
        bootstrap_room=7,
    )


class TestSchedulerIntakeBootstrapHost(CustomTestCase):
    def _intake(self, scheduler, recv_req):
        """Run intake and return the prepared abort's call, or None if the
        request was admitted."""
        with (
            patch(
                "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
                return_value=1,
            ),
            patch(
                "sglang.srt.managers.scheduler.Req",
                return_value=MagicMock(return_logprob=False),
            ),
            patch("sglang.srt.managers.scheduler.prepare_abort") as prepare_abort,
        ):
            try:
                scheduler.handle_generate_request(recv_req)
            except _PastIntake:
                prepare_abort.assert_not_called()
                return None
        prepare_abort.assert_called_once()
        return prepare_abort.call_args

    def test_real_backend_decode_rejects_request_without_bootstrap_host(self):
        scheduler = _scheduler(DisaggregationMode.DECODE, TransferBackend.MOONCAKE)
        abort = self._intake(scheduler, _recv_req(bootstrap_host=None))

        self.assertIsNotNone(abort, "request was admitted")
        self.assertEqual(abort.kwargs["status_code"], HTTPStatus.BAD_REQUEST)
        self.assertIn("without bootstrap host", abort.args[1])
        scheduler.output_streamer.stream_output.assert_called_once()
        scheduler._add_request_to_queue.assert_not_called()

    def test_valid_requests_are_admitted(self):
        cases = [
            # The fake backend needs no prefill peer.
            (DisaggregationMode.DECODE, TransferBackend.FAKE, None),
            (DisaggregationMode.DECODE, TransferBackend.MOONCAKE, PREFILL_HOST),
            # Warm-up and health-check requests.
            (DisaggregationMode.DECODE, TransferBackend.MOONCAKE, FAKE_BOOTSTRAP_HOST),
            # Prefill does not connect to `bootstrap_host`.
            (DisaggregationMode.PREFILL, TransferBackend.MOONCAKE, None),
        ]
        for mode, backend, host in cases:
            with self.subTest(mode=mode, backend=backend, host=host):
                scheduler = _scheduler(mode, backend)
                self.assertIsNone(self._intake(scheduler, _recv_req(host)))


if __name__ == "__main__":
    unittest.main()
