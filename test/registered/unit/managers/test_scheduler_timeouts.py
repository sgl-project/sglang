"""Boundary tests for the scheduler's waiting / running request timeouts.

Both paths are pure bookkeeping over timestamps -- no model, no GPU, no draft
worker -- so they are driven here directly instead of through a server. The
e2e side (503 reaching the client, server stays up) is covered by
scheduler/test_scheduler_control.py.
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.environ import envs
from sglang.srt.observability.metrics_collector import (
    QUEUE_REJECTION_REASON_WAITING_TIMEOUT,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _FakeReq:
    """Must stay hashable: the waiting-timeout path collects drops in a set."""

    def __init__(self, rid, wait_entry=0.0, forward_entry=0.0, is_finished=False):
        self.rid = rid
        self.to_finish = None
        self._finished = is_finished
        self.time_stats = SimpleNamespace(
            wait_queue_entry_time=wait_entry,
            forward_entry_time=forward_entry,
            trace_ctx=MagicMock(),
        )

    def finished(self):
        return self._finished


def _req(
    rid: str, *, wait_entry: float = 0.0, forward_entry: float = 0.0, finished=False
):
    return _FakeReq(rid, wait_entry, forward_entry, finished)


def _scheduler(waiting_queue):
    s = Scheduler.__new__(Scheduler)
    s.waiting_queue = waiting_queue
    s.enable_hicache_storage = False
    s.ps = SimpleNamespace(pp_rank=0)
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    # The waiting-timeout path reports dropped requests to the rejection counter,
    # so the reporter has to exist even though this file asserts nothing about it.
    s.metrics_reporter = SimpleNamespace(
        current_scheduler_metrics_enabled=True, metrics_collector=MagicMock()
    )
    return s


class TestWaitingTimeout(CustomTestCase):
    def test_drops_only_reqs_past_the_deadline(self):
        now = time.perf_counter()
        stale = _req("stale", wait_entry=now - 10)
        fresh = _req("fresh", wait_entry=now)
        s = _scheduler([stale, fresh])

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            s._abort_on_waiting_timeout()

        self.assertEqual([r.rid for r in s.waiting_queue], ["fresh"])
        self.assertEqual(s.ipc_channels.send_to_tokenizer.send_output.call_count, 1)
        collector = s.metrics_reporter.metrics_collector
        collector.increment_queue_rejected_reqs.assert_called_once_with(
            QUEUE_REJECTION_REASON_WAITING_TIMEOUT, 1
        )

    def test_non_reporting_rank_does_not_count_the_drop(self):
        # Requests are broadcast to every TP rank, so each rank runs this path for
        # the same logical request. Ranks without current_scheduler_metrics_enabled
        # must stay quiet or `sum by (reason)` counts one rejection tp_size times.
        s = _scheduler([_req("stale", wait_entry=time.perf_counter() - 10)])
        s.metrics_reporter.current_scheduler_metrics_enabled = False

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            s._abort_on_waiting_timeout()

        self.assertEqual(s.waiting_queue, [], "the req must still be dropped")
        collector = s.metrics_reporter.metrics_collector
        collector.increment_queue_rejected_reqs.assert_not_called()

    def test_nonzero_pp_rank_does_not_count_the_drop(self):
        # Each PP stage can process the same logical request, so queue rejection
        # counters are emitted only from pp_rank 0.
        s = _scheduler([_req("stale", wait_entry=time.perf_counter() - 10)])
        s.ps.pp_rank = 1

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            s._abort_on_waiting_timeout()

        self.assertEqual(s.waiting_queue, [], "the req must still be dropped")
        collector = s.metrics_reporter.metrics_collector
        collector.increment_queue_rejected_reqs.assert_not_called()

    def test_unset_entry_time_is_never_dropped(self):
        # 0 is the "not yet stamped" sentinel; the guard is `0 < entry_time`.
        s = _scheduler([_req("unstamped", wait_entry=0.0)])
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1e-9):
            s._abort_on_waiting_timeout()
        self.assertEqual(len(s.waiting_queue), 1)
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_disabled_timeout_is_a_no_op(self):
        s = _scheduler([_req("stale", wait_entry=time.perf_counter() - 100)])
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(0):
            s._abort_on_waiting_timeout()
        self.assertEqual(len(s.waiting_queue), 1)


class TestRunningTimeout(CustomTestCase):
    @staticmethod
    def _batch(reqs):
        return SimpleNamespace(reqs=reqs, is_empty=lambda: not reqs)

    def test_marks_only_stale_unfinished_reqs(self):
        now = time.perf_counter()
        stale = _req("stale", forward_entry=now - 10)
        fresh = _req("fresh", forward_entry=now)
        done = _req("done", forward_entry=now - 10, finished=True)
        s = _scheduler([])

        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            s._abort_on_running_timeout(self._batch([stale, fresh, done]))

        self.assertIsNotNone(stale.to_finish)
        self.assertIsNone(fresh.to_finish)
        self.assertIsNone(done.to_finish, "a finished req must not be aborted")

    def test_unset_forward_entry_time_is_never_marked(self):
        s = _scheduler([])
        req = _req("unstamped", forward_entry=0.0)
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1e-9):
            s._abort_on_running_timeout(self._batch([req]))
        self.assertIsNone(req.to_finish)

    def test_empty_batch_and_disabled_timeout_are_no_ops(self):
        s = _scheduler([])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            s._abort_on_running_timeout(self._batch([]))
        req = _req("stale", forward_entry=time.perf_counter() - 100)
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(0):
            s._abort_on_running_timeout(self._batch([req]))
        self.assertIsNone(req.to_finish)


if __name__ == "__main__":
    unittest.main()
