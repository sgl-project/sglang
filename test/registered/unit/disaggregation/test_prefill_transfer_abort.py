"""A request abort on a prefill request whose KV send is in flight is not a
transfer failure.

`_abort_prefill_transfer` aborts the request's KV sender, and the sender then
reports a failed transfer, even when the transfer had already succeeded.
`handle_inflight_transfer_failure` logs that failure at debug and does not
count it in the transfer-failure metric. A transfer failure without a request
abort, and a scheduler timeout abort, keep the warning and the count. The
response is the same in every case.
"""

import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_Mixin = SchedulerDisaggregationPrefillMixin


class _Sender:
    """A KV sender that reports an abort as a failed transfer."""

    def __init__(self, state=KVPoll.Transferring):
        self.state = state
        self.reason = "transfer failed" if state == KVPoll.Failed else None
        self.aborted = False
        self.failure_checked = False

    def abort(self):
        self.aborted = True
        self.state = KVPoll.Failed
        self.reason = self.reason or "aborted"

    def failure_exception(self):
        self.failure_checked = True
        raise RuntimeError(self.reason)


def _req(sender_state=KVPoll.Transferring):
    req = Req(
        rid="req-1",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3]),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    req.time_stats.trace_ctx = MagicMock()
    req.bootstrap_room = 7
    req.disagg_kv_sender = _Sender(sender_state)
    return req


class TestPrefillTransferAbort(CustomTestCase):
    def setUp(self):
        self.enterContext(published_topology())
        self.logger = self.enterContext(
            patch("sglang.srt.disaggregation.prefill.logger")
        )

    def _conclude(self, req, abort_req=None):
        scheduler = SimpleNamespace(
            disagg_prefill_transfer_queue=[req],
            request_retirer=MagicMock(),
            metrics_reporter=SimpleNamespace(enable_metrics=True),
            metrics_collector=MagicMock(),
        )
        if abort_req is not None:
            _Mixin._abort_prefill_transfer(scheduler, abort_req)
        _Mixin.handle_inflight_transfer_failure(scheduler, req)
        # The response and the release do not depend on the cause.
        self.assertTrue(req.disagg_kv_sender.failure_checked)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertIn("Prefill transfer failed", req.finished_reason.message)
        self.assertEqual(
            req.finished_reason.status_code, HTTPStatus.INTERNAL_SERVER_ERROR
        )
        scheduler.request_retirer.release.assert_called_once_with(req, checkpoint=False)
        return scheduler.metrics_collector.increment_transfer_failed_reqs

    def _assert_transfer_failure(self, count):
        self.logger.warning.assert_called_once()
        count.assert_called_once_with()

    def test_request_abort_is_not_a_transfer_failure(self):
        req = _req()
        count = self._conclude(req, AbortReq(rid="req-1"))
        self.assertTrue(req.disagg_kv_sender.aborted)
        self.logger.warning.assert_not_called()
        self.assertIn("Prefill transfer failed", self.logger.debug.call_args[0][0])
        count.assert_not_called()

    def test_scheduler_timeout_abort_stays_a_transfer_failure(self):
        req = _req()
        abort = AbortReq(rid="req-1", abort_message="Request running timeout reached.")
        self._assert_transfer_failure(self._conclude(req, abort))
        # The timeout still aborts the sender, so the queue concludes the request.
        self.assertTrue(req.disagg_kv_sender.aborted)

    def test_abort_for_another_request_leaves_it_unmarked(self):
        req = _req()
        scheduler = SimpleNamespace(disagg_prefill_transfer_queue=[req])
        _Mixin._abort_prefill_transfer(scheduler, AbortReq(rid="other"))
        self.assertFalse(req.disagg_kv_sender.aborted)
        self.assertFalse(req.disagg_transfer_aborted)

    def test_transfer_failure_warns_and_counts(self):
        req = _req(sender_state=KVPoll.Failed)
        self._assert_transfer_failure(self._conclude(req))


if __name__ == "__main__":
    unittest.main()
