"""Finished requests leave the running batch before admission on every step (#43961)."""

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
    def __init__(self):
        self.done = False

    def finished(self):
        return self.done


class _Batch:
    def __init__(self, reqs, extend, batch_is_full=False):
        self.reqs = list(reqs)
        self.forward_mode = SimpleNamespace(is_extend=lambda: extend)
        self.chunked_req = None
        self.is_prefill_only = False
        self.batch_is_full = batch_is_full

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, reqs_to_exclude=()):
        self.reqs = [
            r for r in self.reqs if not r.finished() and r not in reqs_to_exclude
        ]

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)


class TestRunningBatchFinished(unittest.TestCase):
    def _scheduler(self, seen):
        s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = None
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
        s.update_running_batch = Mock(side_effect=lambda b: b)
        s._arm_prefill_decode_interval = Mock()
        s.ngram_embedding_manager = Mock()
        s.ngram_embedding_manager.prepare_for_forward.side_effect = lambda b, **_: b

        def get_new_batch_prefill(running_batch):
            seen.append((list(running_batch.reqs), running_batch.batch_is_full))
            # Chunked prefill keeps going: every step runs a prefill, never a decode.
            prefill = _Batch([], extend=True)
            return NextBatchPlan(batch_to_run=prefill, running_batch=running_batch)

        s.get_new_batch_prefill = get_new_batch_prefill
        return s

    def test_request_finished_at_prefill_leaves_before_next_admission(self):
        seen = []
        s = self._scheduler(seen)
        live, short = _Req(), _Req()
        running = _Batch([live], extend=False, batch_is_full=True)

        # Overlap: the prefill that runs `short` merges before its result
        # marks `short` finished.
        plan = s.get_next_batch_to_run(
            running_batch=running, last_batch=_Batch([short], extend=True)
        )
        self.assertEqual(seen[-1], ([live, short], True))

        short.done = True
        s.get_next_batch_to_run(
            running_batch=plan.running_batch, last_batch=plan.batch_to_run
        )
        reqs, batch_is_full = seen[-1]
        self.assertEqual(reqs, [live])
        self.assertFalse(batch_is_full)
        s.update_running_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
