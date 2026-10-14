"""Boundary tests for the scheduler's waiting / running request timeouts.

The poll is pure bookkeeping over timestamps -- no model, no GPU, no draft
worker -- so it is driven here directly instead of through a server. The 503
reaching the client is covered by scheduler/test_scheduler_control.py.
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
)
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeReq:
    release_mm_inputs_on_abort = Req.release_mm_inputs_on_abort

    def __init__(self, rid, wait_entry=0.0, forward_entry=0.0, is_finished=False):
        self.rid = rid
        self.cache_request_handle = CacheRequestHandle(rid, 0)
        self.to_finish = None
        self.beam_group = None
        self.session = None
        self.multimodal_inputs = None
        self._finished = is_finished
        self.output_ids = []
        self.weight_version_events = []
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


def _scheduler(waiting_queue, decoding=(), chunked_req=None):
    s = Scheduler.__new__(Scheduler)
    s.lists = CoreLists()
    s.lists.waiting_queue = waiting_queue
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    s.beam_coordinator = MagicMock()
    s.lists.decoding = list(decoding)
    s.lists.chunked_req = chunked_req
    s.disaggregation_mode = DisaggregationMode.NULL
    s.disagg_prefill = None
    s.disagg_decode = None
    s.grammar_manager = SimpleNamespace(grammar_queue=[], abort_requests=MagicMock())
    s.dllm = None
    s.hisparse_coordinator = None
    return s


class TestQueuedLimitAbort(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())
        patcher = patch(
            "sglang.srt.managers.scheduler.get_serving",
            return_value=SimpleNamespace(weight_version="v0"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_hicache_without_storage_uses_common_abort_cleanup(self):
        candidate = _req("candidate", wait_entry=1.0)
        candidate.priority = 0
        incoming = _req("incoming", wait_entry=2.0)
        incoming.priority = 1

        s = _scheduler([candidate])
        s.max_queued_requests = 1
        get_context().override(
            "test",
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=False,
            enable_hierarchical_cache=True,
        )
        s.tree_cache = MagicMock(spec=["finish"])

        self.assertFalse(s._abort_on_queued_limit(incoming))

        s.tree_cache.finish.assert_called_once_with(
            candidate.cache_request_handle, CacheRequestOutcome.ABORT
        )
        self.assertEqual(s.lists.waiting_queue, [])


class TestWaitingTimeout(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_emits_only_reqs_past_the_deadline_and_keeps_queue_intact(self):
        now = time.perf_counter()
        stale = _req("stale", wait_entry=now - 10)
        fresh = _req("fresh", wait_entry=now)
        s = _scheduler([stale, fresh])

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            aborts = s._poll_timeout_aborts()

        self.assertEqual([a.rid for a in aborts], ["stale"])
        self.assertEqual(aborts[0].finished_reason["type"], "abort")
        # The poll emits only; removal happens on every rank via the broadcast.
        self.assertEqual([r.rid for r in s.lists.waiting_queue], ["stale", "fresh"])

    def test_unset_entry_time_is_never_emitted(self):
        # 0 is the "not yet stamped" sentinel; the guard is `0 < entry_time`.
        s = _scheduler([_req("unstamped", wait_entry=0.0)])
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1e-9):
            self.assertEqual(s._poll_timeout_aborts(), [])

    def test_disabled_timeout_is_a_no_op(self):
        s = _scheduler([_req("stale", wait_entry=time.perf_counter() - 100)])
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(0):
            self.assertEqual(s._poll_timeout_aborts(), [])


class TestRunningTimeout(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_emits_only_stale_unfinished_reqs_without_marking(self):
        now = time.perf_counter()
        stale = _req("stale", forward_entry=now - 10)
        fresh = _req("fresh", forward_entry=now)
        done = _req("done", forward_entry=now - 10, finished=True)
        s = _scheduler([], decoding=[stale, fresh, done])

        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            aborts = s._poll_timeout_aborts()

        self.assertEqual([a.rid for a in aborts], ["stale"])
        # to_finish is set by abort_request() on every rank, not by the poll.
        self.assertIsNone(stale.to_finish)
        self.assertIsNone(fresh.to_finish)
        self.assertIsNone(done.to_finish, "a finished req must not be aborted")

    def test_chunked_request_between_chunks_is_polled(self):
        stale = _req("stale", forward_entry=time.perf_counter() - 10)
        fresh = _req("fresh", forward_entry=time.perf_counter())
        s = _scheduler([], decoding=[fresh], chunked_req=stale)
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            aborts = s._poll_timeout_aborts()
        self.assertEqual([a.rid for a in aborts], ["stale"])

    def test_pending_abort_is_not_emitted_again_while_forward_is_running(self):
        stale = _req("stale", forward_entry=time.perf_counter() - 10)
        fresh = _req("fresh", forward_entry=time.perf_counter())
        s = _scheduler([], decoding=[stale, fresh])
        s.mm_receiver = None
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            aborts = s._poll_timeout_aborts()
            self.assertEqual([abort.rid for abort in aborts], ["stale"])
            # The broadcast abort marks to_finish before the GPU result arrives.
            s.abort_request(aborts[0])
            self.assertIsInstance(stale.to_finish, FINISH_ABORT)
            self.assertIsNone(fresh.to_finish)
            self.assertFalse(stale.finished())
            for _ in range(3):
                self.assertEqual(s._poll_timeout_aborts(), [])

    def test_requests_stay_visible_until_their_commit(self):
        """A request leaves decoding only at the commit that finishes it, so
        the poll and the inflight readers see it while its steps run."""
        now = time.perf_counter()
        older = _req("older", forward_entry=now - 10)
        newer = _req("newer", forward_entry=now - 10)
        s = _scheduler([], decoding=[older, newer])

        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            aborts = s._poll_timeout_aborts()
        # Abort and weight-version updates share this request inventory.
        self.assertEqual(s.collect_inflight_reqs(), [older, newer])
        self.assertEqual([a.rid for a in aborts], ["older", "newer"])

    def test_unset_forward_entry_time_is_never_emitted(self):
        s = _scheduler([], decoding=[_req("unstamped", forward_entry=0.0)])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1e-9):
            self.assertEqual(s._poll_timeout_aborts(), [])

    def test_empty_batch_and_disabled_timeout_are_no_ops(self):
        s = _scheduler([])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            self.assertEqual(s._poll_timeout_aborts(), [])
        stale = _req("stale", forward_entry=time.perf_counter() - 100)
        s = _scheduler([], decoding=[stale])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(0):
            self.assertEqual(s._poll_timeout_aborts(), [])


if __name__ == "__main__":
    unittest.main()
