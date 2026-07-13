"""Unit tests for abort-by-rid-prefix.

Tokenizer side: with prefix=True, the rid is treated as a prefix — the
early-return gate matches any tracked rid starting with it, matching
tokenizer-held (not yet dispatched) requests are resolved locally, and the
AbortReq is forwarded with prefix=True.

Scheduler side: matching is prefix-based (``rid.startswith``) regardless of
the flag, because batch requests derive child rids as ``f"{rid}_{i}"``.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tokenizer_manager(rids=(), tokenizer_worker_num=1) -> TokenizerManager:
    """Create a TokenizerManager with mocked dependencies, bypassing __init__."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm.server_args.tokenizer_worker_num = tokenizer_worker_num
    tm.enable_metrics = False
    tm.rid_to_state = {rid: Mock() for rid in rids}
    tm.send_to_scheduler = MagicMock()
    return tm


def _sent_req(tm) -> AbortReq:
    tm.send_to_scheduler.send_pyobj.assert_called_once()
    return tm.send_to_scheduler.send_pyobj.call_args.args[0]


class TestAbortRequestPrefix(CustomTestCase):
    def test_prefix_match_sends_abort(self):
        tm = _make_tokenizer_manager(rids=["job-1-seq-0", "job-1-seq-1", "other"])
        tm.abort_request(rid="job-1", prefix=True)

        req = _sent_req(tm)
        self.assertEqual(req.rid, "job-1")
        self.assertTrue(req.prefix)
        self.assertFalse(req.abort_all)

    def test_prefix_without_match_is_ignored(self):
        tm = _make_tokenizer_manager(rids=["other-1", "other-2"])
        tm.abort_request(rid="job-1", prefix=True)

        tm.send_to_scheduler.send_pyobj.assert_not_called()

    def test_prefix_requires_full_prefix_not_substring(self):
        tm = _make_tokenizer_manager(rids=["seq-job-1"])
        tm.abort_request(rid="job-1", prefix=True)

        tm.send_to_scheduler.send_pyobj.assert_not_called()

    def test_exact_match_still_works_without_prefix(self):
        tm = _make_tokenizer_manager(rids=["job-1"])
        tm.abort_request(rid="job-1")

        req = _sent_req(tm)
        self.assertEqual(req.rid, "job-1")
        self.assertFalse(req.prefix)

    def test_exact_mode_does_not_prefix_match(self):
        # rid is only a prefix of a tracked request; exact mode must ignore it.
        tm = _make_tokenizer_manager(rids=["job-1-seq-0"])
        tm.abort_request(rid="job-1")

        tm.send_to_scheduler.send_pyobj.assert_not_called()

    def test_empty_rid_is_ignored(self):
        # An empty rid would prefix-match every request on the scheduler.
        tm = _make_tokenizer_manager(rids=["job-1"])
        tm.abort_request(rid="", prefix=True)

        tm.send_to_scheduler.send_pyobj.assert_not_called()

    def test_multi_tokenizer_worker_skips_local_check(self):
        # With >1 tokenizer workers, rid_to_state is not authoritative; the
        # abort must be forwarded even if this worker tracks no matching rid.
        tm = _make_tokenizer_manager(rids=[], tokenizer_worker_num=2)
        tm.abort_request(rid="job-1", prefix=True)

        req = _sent_req(tm)
        self.assertEqual(req.rid, "job-1")
        self.assertTrue(req.prefix)


def _make_state(rid: str) -> ReqState:
    obj = SimpleNamespace(rid=rid, stream=False, return_logprob=False)
    return ReqState([], False, asyncio.Event(), obj, MagicMock())


def _make_tokenizer_manager_with_states(rids) -> TokenizerManager:
    tm = _make_tokenizer_manager()
    tm.rid_to_state = {rid: _make_state(rid) for rid in rids}
    return tm


class TestAbortTokenizerHeldRequests(CustomTestCase):
    """An abort must also cover requests admitted to the tokenizer (rid_to_state
    populated in _init_req_state) but not yet dispatched to the scheduler —
    e.g. parked at the pause gate during a weight update. The scheduler cannot
    match those rids, so abort_request flags them and the dispatch path
    resolves them as aborted instead of sending them."""

    def test_prefix_abort_flags_matching_states(self):
        tm = _make_tokenizer_manager_with_states(
            ["job-1-seq-0", "job-1-seq-1", "other"]
        )
        tm.abort_request(rid="job-1", prefix=True)

        self.assertTrue(tm.rid_to_state["job-1-seq-0"].abort_before_dispatch)
        self.assertTrue(tm.rid_to_state["job-1-seq-1"].abort_before_dispatch)
        self.assertFalse(tm.rid_to_state["other"].abort_before_dispatch)
        # The AbortReq is still forwarded for already-dispatched requests.
        self.assertEqual(_sent_req(tm).rid, "job-1")

    def test_exact_abort_flags_only_exact_state(self):
        tm = _make_tokenizer_manager_with_states(["job-1", "job-1-seq-0"])
        tm.abort_request(rid="job-1")

        self.assertTrue(tm.rid_to_state["job-1"].abort_before_dispatch)
        self.assertFalse(tm.rid_to_state["job-1-seq-0"].abort_before_dispatch)

    def test_abort_all_flags_every_state(self):
        tm = _make_tokenizer_manager_with_states(["a", "b"])
        tm.abort_request(abort_all=True)

        self.assertTrue(tm.rid_to_state["a"].abort_before_dispatch)
        self.assertTrue(tm.rid_to_state["b"].abort_before_dispatch)

    def test_dispatch_resolves_flagged_request_locally(self):
        tm = _make_tokenizer_manager_with_states(["job-1-seq-0"])
        tm.server_args.weight_version = "v0"
        state = tm.rid_to_state["job-1-seq-0"]
        tm.abort_request(rid="job-1", prefix=True)
        tm.send_to_scheduler.reset_mock()

        tm._send_one_request(SimpleNamespace(rid="job-1-seq-0"))

        # Never dispatched; resolved as aborted so _wait_one_response returns.
        tm.send_to_scheduler.send_pyobj.assert_not_called()
        self.assertTrue(state.finished)
        self.assertTrue(state.event.is_set())
        self.assertNotIn("job-1-seq-0", tm.rid_to_state)
        finish_reason = state.out_list[-1]["meta_info"]["finish_reason"]
        self.assertEqual(finish_reason["type"], "abort")

    def test_dispatch_sends_unflagged_request(self):
        tm = _make_tokenizer_manager_with_states(["job-1", "other"])
        tm.abort_request(rid="job-1")
        tm.send_to_scheduler.reset_mock()

        tokenized_obj = MagicMock()
        tokenized_obj.rid = "other"
        with unittest.mock.patch(
            "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
            side_effect=lambda obj: obj,
        ):
            tm._send_one_request(tokenized_obj)

        tm.send_to_scheduler.send_pyobj.assert_called_once_with(tokenized_obj)
        self.assertIn("other", tm.rid_to_state)

    def test_batch_dispatch_filters_flagged_requests(self):
        tm = _make_tokenizer_manager_with_states(["job-1-seq-0", "job-1-seq-1"])
        tm.server_args.weight_version = "v0"
        tm.abort_request(rid="job-1", prefix=True)
        tm.send_to_scheduler.reset_mock()

        tm._send_batch_request(
            [SimpleNamespace(rid="job-1-seq-0"), SimpleNamespace(rid="job-1-seq-1")]
        )

        tm.send_to_scheduler.send_pyobj.assert_not_called()
        self.assertEqual(tm.rid_to_state, {})


class FakeReq:
    def __init__(self, rid: str):
        self.rid = rid
        self.mamba_pool_idx = None
        self.to_finish = None

    def finished(self) -> bool:
        return False


def _make_scheduler(waiting_rids=(), running_rids=(), chunked_rid=None):
    sched = SimpleNamespace()
    sched.chunked_req = FakeReq(chunked_rid) if chunked_rid is not None else None
    sched.waiting_queue = [FakeReq(rid) for rid in waiting_rids]
    sched.enable_hicache_storage = False
    sched.disaggregation_mode = DisaggregationMode.NULL
    sched.grammar_manager = MagicMock()
    sched.running_batch = SimpleNamespace(reqs=[FakeReq(rid) for rid in running_rids])
    sched.cur_batch = None
    sched.ipc_channels = MagicMock()
    return sched


class TestSchedulerAbortMatching(CustomTestCase):
    """Scheduler-side matching semantics for AbortReq (see io_struct.AbortReq:
    always ``rid.startswith``, so batch children ``f"{rid}_{i}"`` are covered)."""

    def test_prefix_abort_isolates_namespaces(self):
        sched = _make_scheduler(
            waiting_rids=["A::1", "A::2", "B::1"],
            running_rids=["A::3", "B::2"],
        )
        Scheduler.abort_request(sched, AbortReq(rid="A::", prefix=True))

        self.assertEqual([req.rid for req in sched.waiting_queue], ["B::1"])
        running = {req.rid: req for req in sched.running_batch.reqs}
        self.assertIsInstance(running["A::3"].to_finish, FINISH_ABORT)
        self.assertIsNone(running["B::2"].to_finish)
        # Every waiting-queue abort echoes back to the tokenizer for cleanup.
        aborted_rids = {
            call.args[0].rid
            for call in sched.ipc_channels.send_to_tokenizer.send_output.call_args_list
        }
        self.assertEqual(aborted_rids, {"A::1", "A::2"})

    def test_matching_is_prefix_based_even_without_prefix_flag(self):
        # Pre-existing scheduler semantics: batch requests derive child rids as
        # f"{rid}_{i}", so an exact-mode abort for the parent must cover them.
        sched = _make_scheduler(waiting_rids=["job-1_0", "job-1_1", "job-2_0"])
        Scheduler.abort_request(sched, AbortReq(rid="job-1", prefix=False))

        self.assertEqual([req.rid for req in sched.waiting_queue], ["job-2_0"])

    def test_chunked_request_is_prefix_matched(self):
        sched = _make_scheduler(chunked_rid="A::9")
        Scheduler.abort_request(sched, AbortReq(rid="A::", prefix=True))

        self.assertIs(sched._pending_chunked_abort_req, sched.chunked_req)

    def test_abort_all_covers_everything(self):
        sched = _make_scheduler(waiting_rids=["A::1"], running_rids=["B::1"])
        Scheduler.abort_request(sched, AbortReq(abort_all=True))

        self.assertEqual(sched.waiting_queue, [])
        self.assertIsInstance(sched.running_batch.reqs[0].to_finish, FINISH_ABORT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
