"""Tests for sglang:num_queue_rejected_requests_total accounting.

The rejection paths are pure bookkeeping over the waiting queue -- no model, no
GPU -- so they are driven directly instead of through a server. The e2e side
(counter visible on /metrics, aggregate matches the number of 503s, and the same
under TP=2) is covered by core/test_request_queue_validation.py and
core/test_request_queue_metrics_tp.py.
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _RecordingCollector:
    """Stands in for SchedulerMetricsCollector, recording reason -> total count."""

    def __init__(self):
        self.counts = {}

    def increment_queue_rejected_reqs(self, reason: str, count: int = 1) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + count


class _FakeReq:
    """Must stay hashable: the waiting-timeout path collects drops in a set."""

    def __init__(self, rid, priority=0, wait_entry=0.0):
        self.rid = rid
        self.priority = priority
        self.to_finish = None
        self.time_stats = SimpleNamespace(
            wait_queue_entry_time=wait_entry,
            forward_entry_time=0.0,
            trace_ctx=MagicMock(),
        )

    def finished(self):
        return False


def _req(rid: str, *, priority: int = 0, wait_entry: float = 0.0):
    return _FakeReq(rid, priority, wait_entry)


def _scheduler(
    waiting_queue,
    *,
    max_queued_requests=1,
    enable_priority_scheduling=False,
    current_scheduler_metrics_enabled=True,
):
    s = Scheduler.__new__(Scheduler)
    s.waiting_queue = waiting_queue
    s.max_queued_requests = max_queued_requests
    s.enable_priority_scheduling = enable_priority_scheduling
    s.schedule_low_priority_values_first = True
    s.enable_hicache_storage = False
    s.enable_hierarchical_cache = False
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    # enable_metrics stays True even on ranks that must not report, which is
    # exactly the TP > 1 case the counter has to stay quiet for.
    s.metrics_reporter = SimpleNamespace(
        enable_metrics=True,
        current_scheduler_metrics_enabled=current_scheduler_metrics_enabled,
        metrics_collector=_RecordingCollector(),
    )
    return s


def _counts(scheduler):
    return scheduler.metrics_reporter.metrics_collector.counts


class TestQueueFullRejectionMetric(CustomTestCase):
    def test_rejecting_the_incoming_req_counts_queue_full_once(self):
        s = _scheduler([_req("queued")])

        aborted_incoming = s._abort_on_queued_limit(_req("incoming"))

        self.assertTrue(aborted_incoming)
        self.assertEqual(_counts(s), {"queue_full": 1})

    def test_room_in_the_queue_counts_nothing(self):
        # Anchors the counter behind the early return, so hoisting the increment
        # out of the rejection branch cannot go unnoticed.
        s = _scheduler([], max_queued_requests=4)

        self.assertFalse(s._abort_on_queued_limit(_req("incoming")))
        self.assertEqual(_counts(s), {})


class TestPriorityPreemptionMetric(CustomTestCase):
    def test_preempting_a_queued_req_counts_priority_preempted(self):
        # Lower number = higher priority, so the incoming req evicts the queued one.
        queued = _req("queued", priority=5)
        s = _scheduler([queued], enable_priority_scheduling=True)

        aborted_incoming = s._abort_on_queued_limit(_req("incoming", priority=0))

        self.assertFalse(aborted_incoming, "the queued req should be the victim")
        self.assertEqual(_counts(s), {"priority_preempted": 1})
        self.assertEqual(s.waiting_queue, [])

    def test_losing_the_priority_contest_counts_queue_full(self):
        # The incoming req is not better than what is queued, so it is shed itself
        # and must be attributed to queue_full rather than to preemption.
        s = _scheduler([_req("queued", priority=0)], enable_priority_scheduling=True)

        aborted_incoming = s._abort_on_queued_limit(_req("incoming", priority=5))

        self.assertTrue(aborted_incoming)
        self.assertEqual(_counts(s), {"queue_full": 1})


class TestWaitingTimeoutMetric(CustomTestCase):
    def test_counts_every_dropped_req(self):
        now = time.perf_counter()
        s = _scheduler(
            [
                _req("stale-1", wait_entry=now - 10),
                _req("stale-2", wait_entry=now - 10),
                _req("fresh", wait_entry=now),
            ]
        )

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            s._abort_on_waiting_timeout()

        self.assertEqual([r.rid for r in s.waiting_queue], ["fresh"])
        self.assertEqual(_counts(s), {"waiting_timeout": 2})


class TestNonReportingRankStaysSilent(CustomTestCase):
    """Regression for the TP > 1 overcount.

    Requests are broadcast to every TP rank, so each rank runs these rejection
    paths for the same logical request. Only ranks where
    current_scheduler_metrics_enabled is true may report, otherwise a
    `sum by (reason)` query counts one rejection tp_size times.

    The gate is a single branch in _record_queue_rejected_req, so one case per
    caller covers it; the reasons themselves are asserted above.
    """

    def test_queue_full_is_not_counted(self):
        s = _scheduler([_req("queued")], current_scheduler_metrics_enabled=False)

        self.assertTrue(s._abort_on_queued_limit(_req("incoming")))
        self.assertEqual(_counts(s), {})

    def test_waiting_timeout_is_not_counted(self):
        s = _scheduler(
            [_req("stale", wait_entry=time.perf_counter() - 10)],
            current_scheduler_metrics_enabled=False,
        )

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            s._abort_on_waiting_timeout()

        self.assertEqual(s.waiting_queue, [], "the req must still be dropped")
        self.assertEqual(_counts(s), {})


if __name__ == "__main__":
    unittest.main()
