"""Boundary tests for the scheduler's waiting / running request timeouts.

The poll is pure bookkeeping over timestamps -- no model, no GPU, no draft
worker -- so it is driven here directly instead of through a server. The 503
reaching the client is covered by scheduler/test_scheduler_control.py.
"""

import time
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeReq:
    release_mm_inputs_on_abort = Req.release_mm_inputs_on_abort
    request_finish = Req.request_finish

    def __init__(self, rid, wait_entry=0.0, forward_entry=0.0, is_finished=False):
        self.rid = rid
        self.cache_request_handle = CacheRequestHandle(rid, 0)
        self.to_finish = None
        self.beam_group = None
        self.session = None
        self.multimodal_inputs = None
        self._finished = is_finished
        self.origin_input_ids = [0]
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


def _batch(reqs):
    return SimpleNamespace(reqs=reqs, is_empty=lambda: not reqs)


def _scheduler(waiting_queue, running_reqs=(), last_batch_reqs=()):
    s = Scheduler.__new__(Scheduler)
    s.enable_continuous_input_polling = False
    s.result_queue = deque()
    s.waiting_queue = waiting_queue
    s.enable_hierarchical_cache = False
    s.enable_hicache_storage = False
    s.enable_unified_cache_external_linker = False
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    s.rust_server = None
    s.output_streamer = MagicMock()
    s.beam_coordinator = MagicMock()
    s.running_batch = _batch(list(running_reqs))
    s.last_batch = _batch(list(last_batch_reqs)) if last_batch_reqs else None
    return s


class TestQueuedLimitAbort(CustomTestCase):
    def setUp(self):
        enter_scope(self, get_context().override_server_args(weight_version="v0"))

    def test_hicache_without_storage_uses_common_abort_cleanup(self):
        candidate = _req("candidate", wait_entry=1.0)
        candidate.priority = 0
        incoming = _req("incoming", wait_entry=2.0)
        incoming.priority = 1

        s = _scheduler([candidate])
        s.max_queued_requests = 1
        s.enable_priority_scheduling = True
        s.schedule_low_priority_values_first = False
        s.enable_hierarchical_cache = True
        s.tree_cache = MagicMock(spec=["finish"])

        self.assertFalse(s._abort_on_queued_limit(incoming))

        s.tree_cache.finish.assert_called_once_with(
            candidate.cache_request_handle, CacheRequestOutcome.ABORT
        )
        self.assertEqual(s.waiting_queue, [])

    def _token_scheduler(self, queue, *, limit=10, priority=False, low_first=False):
        enter_scope(self, get_context().override_server_args(max_queued_tokens=limit))
        s = _scheduler(queue)
        s.max_queued_requests = None
        s.enable_priority_scheduling = priority
        s.schedule_low_priority_values_first = low_first
        s.tree_cache = MagicMock(spec=["finish"])
        return s

    def _token_req(self, rid, tokens, *, priority=0, entry=1.0):
        req = _req(rid, wait_entry=entry)
        req.origin_input_ids = [0] * tokens
        req.priority = priority
        return req

    def test_token_boundary_counts_cached_inputs_but_not_output(self):
        queued = self._token_req("queued", 6)
        queued.prefix_len = 6
        queued.output_ids = [0] * 100
        incoming = self._token_req("incoming", 4)
        s = self._token_scheduler([queued])
        self.assertFalse(s._abort_on_queued_limit(incoming))
        incoming.origin_input_ids.append(0)
        self.assertTrue(s._abort_on_queued_limit(incoming))
        output = s.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        self.assertEqual(output.finished_reason["status_code"], 503)
        self.assertEqual(s.waiting_queue, [queued])

    def test_both_limits_apply_and_removed_requests_release_budget(self):
        queued = self._token_req("queued", 6)
        incoming = self._token_req("incoming", 5)
        s = self._token_scheduler([queued])
        s.max_queued_requests = 2
        self.assertTrue(s._abort_on_queued_limit(incoming))
        incoming.origin_input_ids = [0]
        s.max_queued_requests = 1
        self.assertTrue(s._abort_on_queued_limit(incoming))
        s.waiting_queue.clear()
        self.assertFalse(s._abort_on_queued_limit(incoming))

    def test_priority_evicts_multiple_requests_in_preference_order(self):
        for low_first in (False, True):
            with self.subTest(low_first=low_first):
                direction = 1 if low_first else -1
                old = self._token_req("old", 3, priority=2 * direction, entry=1)
                new = self._token_req("new", 3, priority=2 * direction, entry=2)
                better = self._token_req("better", 3, priority=direction)
                incoming = self._token_req("incoming", 7, priority=0)
                s = self._token_scheduler(
                    [old, better, new], priority=True, low_first=low_first
                )
                self.assertFalse(s._abort_on_queued_limit(incoming))
                self.assertEqual(s.waiting_queue, [better])
                outputs = s.ipc_channels.send_to_tokenizer.send_output.call_args_list
                self.assertEqual([c.args[0].rid for c in outputs], ["new", "old"])
                self.assertEqual(s.tree_cache.finish.call_count, 2)
                self.assertEqual(s.beam_coordinator.retire_group.call_count, 2)

    def test_failed_priority_admission_does_not_evict_any_requests(self):
        for incoming_tokens in (8, 11):
            with self.subTest(incoming_tokens=incoming_tokens):
                worse = self._token_req("worse", 3, priority=-1)
                equal = self._token_req("equal", 4, priority=0)
                incoming = self._token_req("incoming", incoming_tokens, priority=0)
                s = self._token_scheduler([worse, equal], priority=True)
                self.assertTrue(s._abort_on_queued_limit(incoming))
                self.assertEqual(s.waiting_queue, [worse, equal])
                self.assertEqual(s.tree_cache.finish.call_count, 0)

    def test_zero_limits_reject_without_selecting_from_an_empty_queue(self):
        incoming = self._token_req("incoming", 1)
        s = self._token_scheduler([], limit=0, priority=True)
        self.assertTrue(s._abort_on_queued_limit(incoming))
        s.max_queued_requests = 0
        incoming.origin_input_ids = []
        self.assertTrue(s._abort_on_queued_limit(incoming))


class TestWaitingTimeout(CustomTestCase):
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
        self.assertEqual([r.rid for r in s.waiting_queue], ["stale", "fresh"])

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
        s = _scheduler([], running_reqs=[stale, fresh, done])

        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            aborts = s._poll_timeout_aborts()

        self.assertEqual([a.rid for a in aborts], ["stale"])
        # to_finish is set by abort_request() on every rank, not by the poll.
        self.assertIsNone(stale.to_finish)
        self.assertIsNone(fresh.to_finish)
        self.assertIsNone(done.to_finish, "a finished req must not be aborted")

    def test_req_in_both_running_and_last_batch_is_emitted_once(self):
        stale = _req("stale", forward_entry=time.perf_counter() - 10)
        s = _scheduler([], running_reqs=[stale], last_batch_reqs=[stale])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            aborts = s._poll_timeout_aborts()
        self.assertEqual([a.rid for a in aborts], ["stale"])

    def test_pending_abort_is_not_emitted_again_while_forward_is_running(self):
        stale = _req("stale", forward_entry=time.perf_counter() - 10)
        fresh = _req("fresh", forward_entry=time.perf_counter())
        s = _scheduler([], last_batch_reqs=[fresh])
        s.enable_continuous_input_polling = True
        s.result_queue = deque([(_batch([stale]), None), (s.last_batch, None)])
        s.disaggregation_mode = DisaggregationMode.PREFILL
        s.chunked_req = None
        s.mm_receiver = None
        s.dllm_config = None
        s.grammar_manager = MagicMock()
        s.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        s.disagg_prefill_inflight_queue = []
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

    def test_queued_result_remains_visible_after_a_new_forward_launches(self):
        now = time.perf_counter()
        older = _req("older", forward_entry=now - 10)
        newer = _req("newer", forward_entry=now - 10)
        for pp_size in (1, 2):
            with (
                self.subTest(pp_size=pp_size),
                get_parallel().override(pp_size=pp_size),
            ):
                # Polling re-enters intake before last_batch advances to newer.
                last_req = older if pp_size == 1 else newer
                s = _scheduler([], last_batch_reqs=[last_req])
                s.enable_continuous_input_polling = pp_size == 1
                s.running_mbs = [s.running_batch]
                s.mbs = [s.last_batch, _batch([older])]
                if pp_size == 1:
                    s.result_queue = deque(
                        [(_batch([older]), None), (_batch([newer]), None)]
                    )

                with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
                    aborts = s._poll_timeout_aborts()
                    # Abort and weight-version updates share this request inventory.
                    self.assertEqual(s.collect_inflight_reqs(), {older, newer})

                self.assertEqual(
                    [a.rid for a in aborts],
                    ["older", "newer"] if pp_size == 1 else ["newer", "older"],
                )
                self.assertEqual(len(s.result_queue), 2 if pp_size == 1 else 0)

    def test_unset_forward_entry_time_is_never_emitted(self):
        s = _scheduler([], running_reqs=[_req("unstamped", forward_entry=0.0)])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1e-9):
            self.assertEqual(s._poll_timeout_aborts(), [])

    def test_empty_batch_and_disabled_timeout_are_no_ops(self):
        s = _scheduler([])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            self.assertEqual(s._poll_timeout_aborts(), [])
        stale = _req("stale", forward_entry=time.perf_counter() - 100)
        s = _scheduler([], running_reqs=[stale])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(0):
            self.assertEqual(s._poll_timeout_aborts(), [])


if __name__ == "__main__":
    unittest.main()
