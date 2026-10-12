"""Finished running rows leave at the top of the next plan, before admission."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req:
    def __init__(self, rid, finished=False):
        self.rid = rid
        self.done = finished

    def finished(self):
        return self.done


class _Batch:
    """Keeps the row order and the filter and merge rules of ScheduleBatch."""

    def __init__(self, reqs, extend=False):
        self.reqs = list(reqs)
        self.forward_mode = SimpleNamespace(is_extend=lambda: extend)
        self.chunked_req = None
        self.is_prefill_only = False

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, reqs_to_exclude=()):
        self.reqs = [
            r for r in self.reqs if not r.finished() and r not in reqs_to_exclude
        ]

    def merge_batch(self, other):
        self.reqs = self.reqs + other.reqs


def _rids(batch):
    return [r.rid for r in batch.reqs]


class TestLeaveBeforeAdmission(unittest.TestCase):
    def _scheduler(self, admitted=None):
        """A scheduler whose admission records the running rows it sees."""
        s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = None
        s.scripted_scheduler_hook = None
        s.disaggregation_mode = DisaggregationMode.NULL
        s.process_pending_chunked_abort = Mock()
        s._process_hicache_events = Mock()
        s.enable_fpm = False
        s.dllm_config = None
        s.chunked_req = None
        s.enable_hisparse = False
        s.require_mlp_sync = False
        s._should_defer_prefill = Mock(return_value=False)
        s.dp_attn_adapter = Mock()
        s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda b, **_: b
        s.dp_attn_adapter.maybe_convert_decode_to_extend.side_effect = lambda b: b
        s.select_decode_rows = Mock(return_value=True)
        s._arm_prefill_decode_interval = Mock()
        s.ngram_embedding_manager = Mock()
        s.ngram_embedding_manager.prepare_for_forward.side_effect = lambda b, **_: b
        s.seen = []

        def get_new_batch_prefill(running_batch):
            s.seen.append(_rids(running_batch))
            return NextBatchPlan(batch_to_run=admitted, running_batch=running_batch)

        s.get_new_batch_prefill = get_new_batch_prefill
        return s

    def test_admission_sees_only_live_rows(self):
        # A step that admits a prefill never reaches the decode filter, so
        # without the leave the finished rows would hold slots and budget.
        s = self._scheduler(admitted=_Batch([_Req("new")], extend=True))
        running = _Batch(
            [_Req("a"), _Req("b", True), _Req("c"), _Req("d", True), _Req("e")]
        )

        plan = s.get_next_batch_to_run(running_batch=running, last_batch=None)

        self.assertEqual(s.seen, [["a", "c", "e"]])
        self.assertEqual(_rids(plan.running_batch), ["a", "c", "e"])
        s.select_decode_rows.assert_not_called()

    def test_leave_runs_after_the_join(self):
        # Overlap commits the previous step after this plan's join; the leave
        # then drops the old rows that finished, keeping [old survivors, new].
        s = self._scheduler()
        running = _Batch([_Req("a", True), _Req("b")])
        last = _Batch([_Req("c"), _Req("d")], extend=True)

        plan = s.get_next_batch_to_run(running_batch=running, last_batch=last)

        self.assertEqual(s.seen, [["b", "c", "d"]])
        self.assertIs(plan.batch_to_run, running)

        # The join sees the old rows before they leave: a batch whose rows all
        # finished takes the new rows in, it is not replaced at this plan.
        running = _Batch([_Req("a", True)])
        plan = s.get_next_batch_to_run(running_batch=running, last_batch=last)
        self.assertIs(plan.running_batch, running)
        self.assertEqual(_rids(running), ["c", "d"])

    def test_next_joiner_replaces_an_all_finished_batch(self):
        # Every running row finished: the batch empties at this plan, so the
        # next plan adopts the joining batch instead of merging into it.
        new = _Batch([_Req("new")], extend=True)
        s = self._scheduler(admitted=new)
        running = _Batch([_Req("a", True), _Req("b", True)])

        plan = s.get_next_batch_to_run(running_batch=running, last_batch=None)
        self.assertEqual(s.seen, [[]])
        self.assertIs(plan.batch_to_run, new)

        s.get_new_batch_prefill = lambda rb: NextBatchPlan(
            batch_to_run=None, running_batch=rb
        )
        plan = s.get_next_batch_to_run(running_batch=plan.running_batch, last_batch=new)
        self.assertIs(plan.running_batch, new)
        self.assertEqual(_rids(plan.running_batch), ["new"])


if __name__ == "__main__":
    unittest.main()
