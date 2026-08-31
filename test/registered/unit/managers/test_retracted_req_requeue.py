"""Unit tests for how retracted requests are re-queued.

A retracted request is not a new arrival: it was already admitted and has
already paid for a full prefill. It must therefore not be rejected by the
waiting-queue cap (`--max-queued-requests`), which exists to shed *incoming*
load. Churn is bounded instead by `--max-retraction-count`.
"""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRetractedReqRequeue(unittest.TestCase):
    def setUp(self):
        # _make_abort_req reads get_serving().weight_version from the
        # published config, which does not exist for a __new__-built scheduler.
        patcher = patch(
            "sglang.srt.managers.scheduler.get_serving",
            return_value=SimpleNamespace(weight_version="v0"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _new_scheduler(
        self,
        max_queued_requests=None,
        max_retraction_count=None,
        queue_depth=0,
    ) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_priority_scheduling = False
        scheduler.schedule_low_priority_values_first = False
        scheduler.abort_on_priority_when_disabled = False
        scheduler.enable_hicache_storage = False
        scheduler.max_queued_requests = max_queued_requests
        scheduler.max_retraction_count = max_retraction_count
        scheduler.waiting_queue = [
            self._new_req(rid=f"queued-{i}") for i in range(queue_depth)
        ]
        scheduler._prefetch_kvcache = MagicMock()
        scheduler.model_config = SimpleNamespace(num_key_value_heads=8)
        scheduler.ipc_channels = MagicMock()
        scheduler.beam_coordinator = MagicMock()
        scheduler.tree_cache = MagicMock()
        return scheduler

    def _new_req(self, rid="req", retraction_count=0, priority=None):
        req = MagicMock()
        req.rid = rid
        req.priority = priority
        req.retraction_count = retraction_count
        req.output_ids = []
        req.weight_version_events = []
        req.time_stats = MagicMock()
        req.time_stats.trace_ctx = MagicMock()
        return req

    def _sent_abort(self, scheduler):
        """Return the finished_reason of the abort that was sent, or None."""
        calls = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args_list
        if not calls:
            return None
        return calls[-1].args[0].finished_reason

    # --- queue cap must not kill retracted requests -----------------------

    def test_retracted_req_bypasses_full_queue(self):
        """The regression this branch fixes: a full queue used to 503 a
        retracted request, discarding a prefill it had already paid for."""
        scheduler = self._new_scheduler(max_queued_requests=1, queue_depth=1)
        req = self._new_req(rid="retracted", retraction_count=1)

        scheduler._add_request_to_queue(req, is_retracted=True)

        self.assertIn(req, scheduler.waiting_queue)
        self.assertIsNone(self._sent_abort(scheduler))
        req.time_stats.set_retract_time.assert_called_once()

    def test_new_req_still_rejected_when_queue_full(self):
        """The exemption must not weaken admission control for new arrivals."""
        scheduler = self._new_scheduler(max_queued_requests=1, queue_depth=1)
        req = self._new_req(rid="incoming")

        scheduler._add_request_to_queue(req, is_retracted=False)

        self.assertNotIn(req, scheduler.waiting_queue)
        reason = self._sent_abort(scheduler)
        self.assertIsNotNone(reason)
        self.assertEqual(reason["status_code"], HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertIn("queue is full", reason["message"])

    # --- retraction count bounds the churn --------------------------------

    def test_retraction_count_over_limit_fails_fast(self):
        scheduler = self._new_scheduler(max_queued_requests=8, max_retraction_count=3)
        req = self._new_req(rid="thrashing", retraction_count=4)

        scheduler._add_request_to_queue(req, is_retracted=True)

        self.assertNotIn(req, scheduler.waiting_queue)
        reason = self._sent_abort(scheduler)
        self.assertIsNotNone(reason)
        self.assertEqual(reason["status_code"], HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertIn("max-retraction-count", reason["message"])

    def test_retraction_count_at_limit_is_still_admitted(self):
        scheduler = self._new_scheduler(max_queued_requests=8, max_retraction_count=3)
        req = self._new_req(rid="borderline", retraction_count=3)

        scheduler._add_request_to_queue(req, is_retracted=True)

        self.assertIn(req, scheduler.waiting_queue)
        self.assertIsNone(self._sent_abort(scheduler))

    def test_retraction_count_unlimited_by_default(self):
        """Default (None) preserves the legacy unbounded behavior."""
        scheduler = self._new_scheduler(max_queued_requests=8)
        req = self._new_req(rid="veteran", retraction_count=9999)

        scheduler._add_request_to_queue(req, is_retracted=True)

        self.assertIn(req, scheduler.waiting_queue)
        self.assertIsNone(self._sent_abort(scheduler))

    def test_retraction_limit_applies_before_queue_exemption(self):
        """A request over the retraction limit is dropped even though the
        exemption would otherwise let it into a full queue."""
        scheduler = self._new_scheduler(
            max_queued_requests=1, max_retraction_count=2, queue_depth=1
        )
        req = self._new_req(rid="thrashing", retraction_count=5)

        scheduler._add_request_to_queue(req, is_retracted=True)

        self.assertNotIn(req, scheduler.waiting_queue)
        self.assertEqual(len(scheduler.waiting_queue), 1)
        reason = self._sent_abort(scheduler)
        self.assertIn("max-retraction-count", reason["message"])

    # --- overflow stays bounded -------------------------------------------

    def test_retracted_overflow_is_bounded_by_running_batch(self):
        """Exempting retracted requests can only overflow the queue by the
        number of requests that were actually running."""
        max_queued, max_running = 4, 3
        scheduler = self._new_scheduler(
            max_queued_requests=max_queued, queue_depth=max_queued
        )

        for i in range(max_running):
            scheduler._add_request_to_queue(
                self._new_req(rid=f"retracted-{i}", retraction_count=1),
                is_retracted=True,
            )

        self.assertEqual(len(scheduler.waiting_queue), max_queued + max_running)
        self.assertIsNone(self._sent_abort(scheduler))


if __name__ == "__main__":
    unittest.main()