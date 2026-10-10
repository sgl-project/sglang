"""A running batch that DP attention converts to extend stays the running batch."""

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


class _RunningBatch:
    """A full running batch that DP attention converts to extend in place."""

    def __init__(self, num_reqs):
        self.reqs = [SimpleNamespace(finished=False) for _ in range(num_reqs)]
        self.forward_mode = SimpleNamespace(is_extend=lambda: False)
        self.chunked_req = None
        self.is_prefill_only = False
        self.batch_is_full = True

    def convert_decode_to_extend(self):
        self.forward_mode = SimpleNamespace(is_extend=lambda: True)
        return self

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, chunked_req_to_exclude=()):
        self.reqs = [
            r for r in self.reqs if not r.finished and r not in chunked_req_to_exclude
        ]

    def merge_batch(self, other):
        raise AssertionError("a converted running batch must not be merged")


class TestDpConvertedBatchFull(unittest.TestCase):
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
        s.dp_attn_adapter.maybe_convert_decode_to_extend.side_effect = lambda b: (
            b.convert_decode_to_extend() if b is not None else b
        )
        s.update_running_batch = Mock(side_effect=lambda b: b)
        s._arm_prefill_decode_interval = Mock()
        s.ngram_embedding_manager = Mock()
        s.ngram_embedding_manager.prepare_for_forward.side_effect = lambda b, **_: b

        def get_new_batch_prefill(running_batch):
            seen.append((running_batch, running_batch.batch_is_full))
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

        s.get_new_batch_prefill = get_new_batch_prefill
        return s

    def test_converted_batch_stays_running_and_reopens_admission(self):
        seen = []
        s = self._scheduler(seen)
        running = _RunningBatch(num_reqs=3)

        # Step N: decode is converted to extend because a peer rank extends.
        plan = s.get_next_batch_to_run(running_batch=running, last_batch=None)
        self.assertIs(plan.batch_to_run, running)
        self.assertIs(plan.running_batch, running)

        # Step N+1: one row finished; the batch comes back as last_batch.
        running.reqs[0].finished = True
        plan = s.get_next_batch_to_run(
            running_batch=plan.running_batch, last_batch=plan.batch_to_run
        )

        batch, batch_is_full = seen[-1]
        self.assertIs(batch, running)
        self.assertEqual(running.batch_size(), 2)
        self.assertFalse(batch_is_full)


if __name__ == "__main__":
    unittest.main()
