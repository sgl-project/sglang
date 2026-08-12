"""PP disagg prefill: aborts and KV failures must flush via consensus, never locally.

A KV-layer failure (e.g. a decode-side abort notification) lands on each PP
rank at an arbitrary wall-clock time. If a rank unilaterally deletes such a
request from its bootstrap queue, the deletion races against an in-flight
good consensus and the per-rank waiting queues (and therefore the micro-batch
streams) diverge — downstream ranks then wait forever on a batch that was
never formed. These tests pin the invariant: local signals only mark; queue
mutation happens exclusively at the consensus commit step.
"""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler_pp_mixin import (
    PPBootstrapDecision,
    SchedulerPPMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_POLL_HELPER = "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group"


def _req(rid: str):
    return SimpleNamespace(
        rid=rid,
        bootstrap_room=123,
        finished_reason=None,
        disagg_kv_sender=MagicMock(),
        prefill_attempt_count=1,
        is_retracted=False,
        return_logprob=False,
        time_stats=SimpleNamespace(set_wait_queue_entry_time=MagicMock()),
        to_abort_message=None,
        to_finish=None,
    )


def _make_queue(reqs, *, pp_size: int = 8):
    queue = object.__new__(PrefillBootstrapQueue)
    queue.queue = list(reqs)
    queue.pp_size = pp_size
    queue.scheduler = SimpleNamespace(
        attn_cp_cpu_group=None,
        attn_tp_cpu_group=None,
        handle_bootstrap_failure=MagicMock(),
        _pp_record_pending_bootstrap_failure=MagicMock(),
        server_args=SimpleNamespace(optimistic_prefill_attempts=0),
    )
    queue.finalize_bootstrap = MagicMock(return_value=True)
    queue.ensure_metadata_buffer = MagicMock(return_value=True)
    return queue


class TestPopBootstrappedLocalFailure(unittest.TestCase):
    def test_uncovered_local_failure_marks_but_keeps_request(self):
        """A locally-observed KV failure must not delete the req unilaterally."""
        req = _req("req-a")
        queue = _make_queue([req])

        with patch(_POLL_HELPER, return_value=[KVPoll.Failed]):
            good, failed = queue.pop_bootstrapped(
                return_failed_reqs=True, pp_good_rids=[], pp_bad_rids=[]
            )

        self.assertEqual(good, [])
        self.assertEqual(failed, [])
        self.assertEqual(queue.queue, [req])  # still queued
        self.assertIsNone(req.finished_reason)
        queue.scheduler._pp_record_pending_bootstrap_failure.assert_called_once_with(
            req, "KV transfer failed before bootstrap consensus."
        )
        queue.scheduler.handle_bootstrap_failure.assert_not_called()

    def test_uncovered_local_failure_does_not_remark_aborted_request(self):
        req = _req("req-a")
        original = FINISH_ABORT(message="Aborted by AbortReq.")
        req.finished_reason = original
        queue = _make_queue([req])

        with patch(_POLL_HELPER, return_value=[KVPoll.Failed]):
            queue.pop_bootstrapped(
                return_failed_reqs=True, pp_good_rids=[], pp_bad_rids=[]
            )

        self.assertIs(req.finished_reason, original)
        queue.scheduler._pp_record_pending_bootstrap_failure.assert_not_called()

    def test_uncovered_nonterminal_states_wait_for_a_later_consensus(self):
        for local_poll in (KVPoll.Bootstrapping, KVPoll.WaitingForInput):
            with self.subTest(local_poll=local_poll):
                req = _req("req-a")
                queue = _make_queue([req])

                with patch(_POLL_HELPER, return_value=[local_poll]):
                    good, failed = queue.pop_bootstrapped(
                        return_failed_reqs=True,
                        pp_good_rids=[],
                        pp_bad_rids=[],
                    )

                self.assertEqual(good, [])
                self.assertEqual(failed, [])
                self.assertEqual(queue.queue, [req])
                self.assertIsNone(req.finished_reason)

    def test_uncovered_post_bootstrap_state_fails_loudly(self):
        for local_poll in (KVPoll.Transferring, KVPoll.Success):
            with self.subTest(local_poll=local_poll):
                req = _req("req-a")
                queue = _make_queue([req])

                with patch(_POLL_HELPER, return_value=[local_poll]):
                    with self.assertRaisesRegex(
                        RuntimeError, "Unexpected local KV poll state"
                    ):
                        queue.pop_bootstrapped(
                            return_failed_reqs=True,
                            pp_good_rids=[],
                            pp_bad_rids=[],
                        )

    def test_consensus_bad_flushes_request_at_commit(self):
        """Deletion happens only when the finalized bad list arrives."""
        req = _req("req-a")
        queue = _make_queue([req])

        good, failed = queue.pop_bootstrapped(
            return_failed_reqs=True, pp_good_rids=[], pp_bad_rids=["req-a"]
        )

        self.assertEqual(good, [])
        self.assertEqual(failed, [req])
        self.assertEqual(queue.queue, [])
        queue.scheduler.handle_bootstrap_failure.assert_called_once_with(req)

    def test_consensus_good_admission_cannot_be_vetoed_locally(self):
        """A rid the consensus admitted must be admitted (or crash loudly)."""
        req = _req("req-a")
        queue = _make_queue([req])
        queue.finalize_bootstrap = MagicMock(return_value=False)

        with self.assertRaisesRegex(RuntimeError, "consensus admitted"):
            queue.pop_bootstrapped(
                return_failed_reqs=True, pp_good_rids=["req-a"], pp_bad_rids=[]
            )


class TestBootstrappedIdsConsensusRound(unittest.TestCase):
    def _make_mixin(self, reqs, *, pp_rank: int = 0):
        scheduler = SchedulerPPMixin()
        scheduler.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=8)
        scheduler.pp_group = SimpleNamespace(is_first_rank=pp_rank == 0)
        scheduler.enable_hicache_storage = False
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            queue=list(reqs),
            ensure_metadata_buffer=MagicMock(return_value=True),
        )
        return scheduler

    def test_marked_request_routes_to_bad_union_next_round(self):
        """The mark left by pop_bootstrapped enters bad consensus next round."""
        marked = _req("req-a")
        marked.finished_reason = FINISH_ABORT(
            message="KV transfer failed before bootstrap consensus.",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        clean = _req("req-b")
        scheduler = self._make_mixin([marked, clean])
        scheduler.get_rids = MagicMock(return_value=(["req-a", "req-b"], []))

        good, bad = scheduler._pp_pd_get_bootstrapped_ids()

        self.assertEqual(good, ["req-b"])
        self.assertEqual(bad, ["req-a"])

    def test_pending_local_failure_routes_to_bad_after_request_left_queue(self):
        scheduler = self._make_mixin([])
        scheduler.get_rids = MagicMock(return_value=([], []))
        scheduler._pp_pending_bootstrap_failures = {"req-a": "local KV failure"}

        good, bad = scheduler._pp_pd_get_bootstrapped_ids()

        self.assertEqual(good, [])
        self.assertEqual(bad, ["req-a"])

    def test_metadata_buffer_reserved_before_reporting_good(self):
        """A rank without buffer headroom withholds the rid instead of
        vetoing at commit time."""
        req_a, req_b = _req("req-a"), _req("req-b")
        scheduler = self._make_mixin([req_a, req_b])
        scheduler.get_rids = MagicMock(return_value=(["req-a", "req-b"], []))
        scheduler.disagg_prefill_bootstrap_queue.ensure_metadata_buffer = MagicMock(
            side_effect=lambda req: req.rid != "req-b"
        )

        good, bad = scheduler._pp_pd_get_bootstrapped_ids()

        self.assertEqual(good, ["req-a"])
        self.assertEqual(bad, [])


class TestBootstrapDecisionSequencing(unittest.TestCase):
    def test_bootstrap_decision_satisfies_pp_transport_contract(self):
        decision = PPBootstrapDecision(7, ("good",), ("bad",))

        self.assertEqual(len(decision), 3)
        self.assertEqual(decision.sequence, 7)
        with self.assertRaises(AttributeError):
            decision.sequence = 8

    def _make_scheduler(self, reqs, *, pp_rank: int):
        scheduler = SchedulerPPMixin()
        scheduler.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=8)
        scheduler.pp_group = SimpleNamespace(
            is_first_rank=pp_rank == 0, is_last_rank=pp_rank == 7
        )
        scheduler.disagg_prefill_bootstrap_queue = _make_queue(reqs)
        scheduler.waiting_queue = []
        return scheduler

    def test_pp0_fences_abort_before_freezing_decision(self):
        aborted = _req("req-a")
        aborted.finished_reason = FINISH_ABORT(message="Aborted by AbortReq.")
        clean = _req("req-b")
        scheduler = self._make_scheduler([aborted, clean], pp_rank=0)

        decision = scheduler.process_bootstrapped_queue([["req-a", "req-b"], []])

        self.assertEqual(
            decision,
            PPBootstrapDecision(
                sequence=0,
                good_rids=("req-b",),
                bad_rids=("req-a",),
            ),
        )
        self.assertEqual(scheduler.waiting_queue, [clean])
        self.assertEqual(scheduler.disagg_prefill_bootstrap_queue.queue, [])
        scheduler.disagg_prefill_bootstrap_queue.scheduler.handle_bootstrap_failure.assert_called_once_with(
            aborted
        )

    def test_non_first_rank_applies_and_forwards_exact_decision(self):
        req = _req("req-a")
        scheduler = self._make_scheduler([req], pp_rank=3)
        decision = PPBootstrapDecision(0, ("req-a",), ())

        forwarded = scheduler.process_bootstrapped_queue(decision)

        self.assertIs(forwarded, decision)
        self.assertEqual(scheduler.waiting_queue, [req])
        self.assertEqual(
            scheduler._pp_bootstrap_expected_decision_sequence,
            1,
        )

    def test_duplicate_decision_is_not_replayed(self):
        scheduler = self._make_scheduler([], pp_rank=2)
        queue = scheduler.disagg_prefill_bootstrap_queue
        queue.pop_bootstrapped = MagicMock(return_value=([], []))
        decision = PPBootstrapDecision(0, (), ())

        self.assertIs(scheduler.process_bootstrapped_queue(decision), decision)
        self.assertIs(scheduler.process_bootstrapped_queue(decision), decision)

        queue.pop_bootstrapped.assert_called_once()

    def test_sequence_gap_fails_loudly(self):
        scheduler = self._make_scheduler([], pp_rank=4)

        with self.assertRaisesRegex(RuntimeError, "sequence gap"):
            scheduler.process_bootstrapped_queue(PPBootstrapDecision(1, (), ()))

    def test_local_abort_cannot_override_frozen_good_decision(self):
        req = _req("req-a")
        req.finished_reason = FINISH_ABORT(message="Aborted by AbortReq.")
        scheduler = self._make_scheduler([req], pp_rank=5)

        with self.assertRaisesRegex(RuntimeError, "ordering violation"):
            scheduler.process_bootstrapped_queue(PPBootstrapDecision(0, ("req-a",), ()))

        self.assertEqual(
            scheduler.disagg_prefill_bootstrap_queue.queue,
            [req],
        )

    def test_late_local_kv_failure_does_not_veto_frozen_good_decision(self):
        req = _req("req-a")
        scheduler = self._make_scheduler([req], pp_rank=5)
        scheduler._pp_pending_bootstrap_failures = {"req-a": "local KV failure"}

        scheduler.process_bootstrapped_queue(PPBootstrapDecision(0, ("req-a",), ()))

        self.assertEqual(scheduler.waiting_queue, [req])
        self.assertEqual(
            scheduler._pp_pending_bootstrap_failures,
            {"req-a": "local KV failure"},
        )

    def test_later_bad_commit_aborts_rid_after_good_moved_it_out_of_bootstrap(self):
        req = _req("req-a")
        scheduler = self._make_scheduler([req], pp_rank=5)
        scheduler.abort_request = MagicMock()
        scheduler._pp_pending_bootstrap_failures = {"req-a": "local KV failure"}

        scheduler.process_bootstrapped_queue(PPBootstrapDecision(0, ("req-a",), ()))
        scheduler.process_bootstrapped_queue(PPBootstrapDecision(1, (), ("req-a",)))

        scheduler.abort_request.assert_called_once_with(
            AbortReq(rid="req-a", pp_bootstrap_abort_after_sequence=1)
        )
        self.assertEqual(scheduler._pp_pending_bootstrap_failures, {})

    def test_pp0_stamps_abort_after_last_frozen_decision(self):
        scheduler = self._make_scheduler([], pp_rank=0)
        scheduler.process_bootstrapped_queue([[], []])
        abort_req = AbortReq(rid="req-a")

        deferred = scheduler._pp_order_or_defer_abort_request(abort_req)

        self.assertFalse(deferred)
        self.assertEqual(abort_req.pp_bootstrap_abort_after_sequence, 0)

    def test_downstream_defers_late_abort_until_boundary_is_committed(self):
        req = _req("req-a")
        scheduler = self._make_scheduler([req], pp_rank=3)
        scheduler.abort_request = MagicMock()
        abort_req = AbortReq(
            rid="req-a",
            pp_bootstrap_abort_after_sequence=0,
        )

        deferred = scheduler._pp_order_or_defer_abort_request(abort_req)

        self.assertTrue(deferred)
        scheduler.abort_request.assert_not_called()
        self.assertEqual(
            scheduler.disagg_prefill_bootstrap_queue.queue,
            [req],
        )

        scheduler.process_bootstrapped_queue(PPBootstrapDecision(0, ("req-a",), ()))

        self.assertEqual(scheduler.waiting_queue, [req])
        scheduler.abort_request.assert_called_once_with(abort_req)
        self.assertEqual(list(scheduler._pp_deferred_abort_reqs), [])

    def test_stale_good_after_bad_commit_cannot_resurrect_request(self):
        req = _req("req-a")
        req.finished_reason = FINISH_ABORT(message="Aborted by AbortReq.")
        scheduler = self._make_scheduler([req], pp_rank=0)

        bad_decision = scheduler.process_bootstrapped_queue([["req-a"], []])
        stale_good_decision = scheduler.process_bootstrapped_queue([["req-a"], []])

        self.assertEqual(bad_decision.sequence, 0)
        self.assertEqual(bad_decision.bad_rids, ("req-a",))
        self.assertEqual(stale_good_decision.sequence, 1)
        self.assertEqual(stale_good_decision.good_rids, ("req-a",))
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(scheduler.disagg_prefill_bootstrap_queue.queue, [])


if __name__ == "__main__":
    unittest.main()
