"""Unit tests for PP grammar ready consensus state transitions."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerGrammarNonPPGate(CustomTestCase):
    def test_pp_scheduler_does_not_pop_grammar_queue_without_consensus(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = SimpleNamespace(pp_size=2)
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = True
        scheduler._add_request_to_queue = MagicMock()

        Scheduler._process_ready_grammar_requests_non_pp(scheduler)

        scheduler.grammar_manager.get_ready_grammar_requests.assert_not_called()
        scheduler._add_request_to_queue.assert_not_called()


class TestSchedulerPPGrammarConsensusHelpers(CustomTestCase):
    def _new_mixin(self, *, is_first_rank=False, is_last_rank=False):
        mixin = SchedulerPPMixin()
        mixin.pp_group = MagicMock()
        mixin.pp_group.is_first_rank = is_first_rank
        mixin.pp_group.is_last_rank = is_last_rank
        return mixin

    def test_merge_grammar_ready_ids_intersects_ready_and_unions_failed(self):
        mixin = self._new_mixin()

        merged = mixin._pp_merge_grammar_ready_ids(
            [["r0", "r1", "r2"], ["f0"]],
            [["r2", "r0", "local-only"], ["f1", "f0"]],
        )

        self.assertEqual(merged, [["r2", "r0"], ["f0", "f1"]])

    def test_first_rank_returns_empty_consensus_when_queue_is_nonempty(self):
        mixin = self._new_mixin(is_first_rank=True)
        mixin.grammar_manager = MagicMock()
        mixin.grammar_manager.has_waiting_grammars.return_value = True
        mixin.grammar_manager.poll_ready_grammar_request_rids.return_value = [
            [],
            [],
        ]

        self.assertEqual(mixin._pp_get_grammar_ready_ids(), [[], []])

    def test_non_first_rank_receives_upstream_even_when_local_poll_is_empty(self):
        mixin = self._new_mixin(is_first_rank=False)
        mixin.grammar_manager = MagicMock()
        mixin.grammar_manager.has_waiting_grammars.return_value = True
        mixin.grammar_manager.poll_ready_grammar_request_rids.return_value = [
            [],
            [],
        ]
        mixin._pp_recv_pyobj_from_prev_stage = MagicMock(
            return_value=[["upstream-ready"], ["upstream-failed"]]
        )

        self.assertEqual(mixin._pp_get_grammar_ready_ids(), [[], ["upstream-failed"]])
        mixin._pp_recv_pyobj_from_prev_stage.assert_called_once()

    def test_empty_ready_failed_payload_counts_as_inflight_consensus(self):
        mixin = self._new_mixin()

        self.assertFalse(mixin._pp_has_inflight_grammar_consensus([None, None]))
        self.assertTrue(mixin._pp_has_inflight_grammar_consensus([None, [[], []]]))

    def test_last_rank_emits_consensus_only_when_slot_is_ready(self):
        mixin = self._new_mixin(is_last_rank=True)
        mixin._pp_send_consensus_grammar_ids = MagicMock(return_value=["work"])

        work, consensus = mixin._pp_pd_send_consensus_grammar_ids(
            [None, [["rid-1"], []]], 0, None
        )
        self.assertEqual(work, [])
        self.assertIsNone(consensus)
        mixin._pp_send_consensus_grammar_ids.assert_not_called()

        work, consensus = mixin._pp_pd_send_consensus_grammar_ids(
            [None, [["rid-1"], []]], 1, None
        )
        self.assertEqual(work, ["work"])
        self.assertEqual(consensus, [["rid-1"], []])
        mixin._pp_send_consensus_grammar_ids.assert_called_once_with([["rid-1"], []])

    def test_non_last_rank_forwards_prior_consensus(self):
        mixin = self._new_mixin(is_last_rank=False)
        mixin._pp_send_consensus_grammar_ids = MagicMock(return_value=["work"])

        work, consensus = mixin._pp_pd_send_consensus_grammar_ids(
            [[["local-slot"], []]], 0, [["prior"], ["failed"]]
        )

        self.assertEqual(work, ["work"])
        self.assertEqual(consensus, [["prior"], ["failed"]])
        mixin._pp_send_consensus_grammar_ids.assert_called_once_with(
            [["prior"], ["failed"]]
        )

    def test_non_last_rank_does_not_send_without_prior_consensus(self):
        mixin = self._new_mixin(is_last_rank=False)
        mixin._pp_send_consensus_grammar_ids = MagicMock(return_value=["work"])

        work, consensus = mixin._pp_pd_send_consensus_grammar_ids(
            [[["local-slot"], []]], 0, None
        )

        self.assertEqual(work, [])
        self.assertIsNone(consensus)
        mixin._pp_send_consensus_grammar_ids.assert_not_called()

    def test_duplicate_final_consensus_is_idempotent_at_consumer(self):
        mixin = self._new_mixin()
        req = MagicMock()
        req.rid = "rid-0"
        mixin.grammar_manager = MagicMock()
        mixin.grammar_manager.pop_ready_grammar_requests_by_rids.side_effect = [
            [req],
            [],
        ]
        mixin._add_request_to_queue = MagicMock()

        consensus = [["rid-0"], []]
        mixin.process_grammar_queue(consensus)
        mixin.process_grammar_queue(consensus)

        self.assertEqual(
            mixin.grammar_manager.pop_ready_grammar_requests_by_rids.call_count, 2
        )
        mixin._add_request_to_queue.assert_called_once_with(req)


if __name__ == "__main__":
    unittest.main()
