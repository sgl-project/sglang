"""A prefill server answers a request with no bootstrap host by itself.

The transfer-queue release path finishes such a request through
``update_finish_state``, so the response carries the real finish reason;
a request with a bootstrap host, the fake-bootstrap sentinel, or any
request on the fake transfer backend still takes the handoff finish.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation import utils  # noqa: E402
from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.mode import (  # noqa: E402
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
)
from sglang.srt.disaggregation.prefill import (  # noqa: E402
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import (  # noqa: E402
    ReqToMetadataIdxAllocator,
    TransferBackend,
    answered_by_prefill,
    lacks_decode_peer,
)
from sglang.srt.managers.req_types import FINISH_LENGTH  # noqa: E402
from sglang.srt.managers.scheduler_components.request_list import (  # noqa: E402
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.runtime_context import get_parallel  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.scheduler_test_utils import FakeReq, leaving_retirer  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PREFILL = DisaggregationMode.PREFILL


def _req(host, max_new_tokens=1):
    return SimpleNamespace(
        rid="r",
        bootstrap_host=host,
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
    )


class TestAnsweredByPrefill(CustomTestCase):
    def test_only_a_hostless_request_on_a_real_backend_prefill_server(self):
        cases = [
            (PREFILL, TransferBackend.NIXL, None, True),
            (PREFILL, TransferBackend.MOONCAKE, None, True),
            (PREFILL, TransferBackend.NIXL, "10.0.0.1", False),
            (PREFILL, TransferBackend.NIXL, FAKE_BOOTSTRAP_HOST, False),
            (PREFILL, TransferBackend.FAKE, None, False),
            (DisaggregationMode.DECODE, TransferBackend.NIXL, None, False),
            (DisaggregationMode.NULL, TransferBackend.MOONCAKE, None, False),
        ]
        for mode, backend, host, expected in cases:
            with self.subTest(mode=mode, backend=backend, host=host):
                kwargs = dict(disaggregation_mode=mode, transfer_backend=backend)
                self.assertEqual(lacks_decode_peer(_req(host), **kwargs), expected)
                self.assertEqual(answered_by_prefill(_req(host), **kwargs), expected)

    def test_admitted_request_asks_for_at_most_one_token(self):
        kwargs = dict(
            disaggregation_mode=PREFILL, transfer_backend=TransferBackend.NIXL
        )
        for asked in (0, 1):
            self.assertTrue(answered_by_prefill(_req(None, asked), **kwargs))
        # Intake rejects these before admission; reaching here is a bug.
        for asked in (2, 128, None):
            with self.assertRaises(AssertionError):
                answered_by_prefill(_req(None, asked), **kwargs)


class _Stage(SchedulerDisaggregationPrefillMixin):
    """Single-rank prefill scheduler running the real transfer-queue path."""

    def __init__(self, transfer_backend):
        self.disaggregation_mode = PREFILL
        self.transfer_backend = transfer_backend
        self.attn_cp_cpu_group = self.attn_tp_cpu_group = None
        self.disagg_prefill_transfer_queue = IndexedReqList(ListKind.PD_SENDING)
        self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(4)
        self.request_retirer = leaving_retirer()
        self.metrics_reporter = SimpleNamespace(enable_metrics=True)
        self.metrics_collector = Mock()
        self.batch_result_processor = Mock()
        self.scheduler_stage_metrics = None
        self.streamed = []
        self.output_streamer = SimpleNamespace(
            stream_output=lambda reqs, *args: self.streamed.extend(reqs)
        )
        self._release_mm_inputs = lambda req: None

    def transfer(self, host, max_new_tokens=1):
        """Queue a request whose transfer concludes at the next poll, recording
        which finish path the release takes."""
        calls = []
        req = FakeReq(
            rid="r",
            bootstrap_room=7,
            bootstrap_host=host,
            metadata_buffer_index=self.req_to_metadata_buffer_idx_allocator.alloc(),
            pending_bootstrap=False,
            finished_reason=None,
            return_logprob=False,
            return_routed_experts=False,
            sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
            calls=calls,
            disagg_kv_sender=SimpleNamespace(
                poll=lambda: KVPoll.Success,
                clear=lambda: None,
                get_transfer_metric=lambda: None,
                kv_mgr=None,
            ),
            time_stats=SimpleNamespace(
                set_prefill_kv_transfer_finish_time=lambda: None,
                set_completion_time=lambda: None,
                compute_and_observe_kv_transfer_metrics=lambda metric: None,
                trace_ctx=SimpleNamespace(abort=lambda **kw: None),
            ),
        )

        def finish_outside_step(reason):
            calls.append(("outside", reason))
            req.finished_reason = reason

        def update_finish_state(new_accepted_len=1):
            calls.append(("update", None))
            req.finished_reason = "the sampled token's own finish"

        req.finish_outside_step = finish_outside_step
        req.update_finish_state = update_finish_state
        req.finished = lambda: req.finished_reason is not None
        move([req], self.disagg_prefill_transfer_queue)
        return req

    def release(self):
        with get_parallel().override(pp_rank=0, pp_size=1, tp_rank=0):
            return self.process_disagg_prefill_transfer_queue()


class TestTransferQueueFinish(CustomTestCase):
    def setUp(self):
        self._patch = patch.object(
            utils, "_all_reduce_polls", lambda polls, group: polls
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def _finish(self, backend, host, max_new_tokens=1):
        stage = _Stage(backend)
        req = stage.transfer(host, max_new_tokens)
        done = stage.release()
        self.assertEqual(done, [req])
        self.assertEqual(stage.streamed, [req])
        self.assertEqual(len(req.calls), 1)
        return req.calls[0]

    def test_hostless_request_finishes_with_its_real_reason(self):
        for asked in (0, 1):
            with self.subTest(max_new_tokens=asked):
                kind, _ = self._finish(
                    TransferBackend.NIXL, host=None, max_new_tokens=asked
                )
                self.assertEqual(kind, "update")

    def test_hosted_request_still_hands_off(self):
        kind, reason = self._finish(TransferBackend.NIXL, host="10.0.0.1")
        self.assertEqual(kind, "outside")
        self.assertIsInstance(reason, FINISH_LENGTH)
        self.assertEqual(reason.length, 0)

    def test_sentinel_request_still_hands_off(self):
        kind, reason = self._finish(TransferBackend.NIXL, host=FAKE_BOOTSTRAP_HOST)
        self.assertEqual(kind, "outside")
        self.assertEqual(reason.length, 0)

    def test_fake_backend_hostless_request_still_hands_off(self):
        kind, reason = self._finish(TransferBackend.FAKE, host=None)
        self.assertEqual(kind, "outside")
        self.assertEqual(reason.length, 0)


if __name__ == "__main__":
    unittest.main()
