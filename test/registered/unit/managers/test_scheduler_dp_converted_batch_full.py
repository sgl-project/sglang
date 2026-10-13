"""DP attention converts a decode step to extend; the running rows keep decoding."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.decoding_batch import DecodingBatch

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_DECODE = SimpleNamespace(is_extend=lambda: False)
_EXTEND = SimpleNamespace(is_extend=lambda: True)


class _RunningBatch:
    """Running rows whose decode step DP attention converts to extend."""

    def __init__(self, num_reqs):
        self.reqs = [SimpleNamespace(finished=False) for _ in range(num_reqs)]
        self.forward_mode = _DECODE
        self.num_decode_rows = 0
        self.chunked_req = None
        self.is_prefill_only = False

    def shallow_copy(self):
        batch = copy.copy(self)
        batch.reqs = self.reqs[:]
        return batch

    def convert_decode_to_extend(self):
        self.forward_mode = _EXTEND
        self.num_decode_rows = len(self.reqs)
        return self

    def has_prefill_rows(self):
        return self.forward_mode.is_extend() and self.num_decode_rows < len(self.reqs)

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, reqs_to_exclude=()):
        self.reqs = [
            r for r in self.reqs if not r.finished and r not in reqs_to_exclude
        ]

    def merge_batch(self, other):
        raise AssertionError("a converted running batch must not be merged")


class TestDpConvertedBatchFull(unittest.TestCase):
    def _scheduler(self, seen):
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
        s.dp_attn_adapter.maybe_convert_decode_to_extend.side_effect = lambda b: (
            b.convert_decode_to_extend() if b is not None else b
        )
        s.select_decode_rows = Mock(return_value=True)
        s._arm_prefill_decode_interval = Mock()
        s.ngram_embedding_manager = Mock()
        s.ngram_embedding_manager.prepare_for_forward.side_effect = lambda b, **_: b

        def get_new_batch_prefill(decoding_batch):
            seen.append(decoding_batch.rows)
            return NextBatchPlan(batch_to_run=None, running_batch=decoding_batch.rows)

        s.get_new_batch_prefill = get_new_batch_prefill
        return s

    def test_rows_unchanged_by_conversion_and_admit_next_step(self):
        seen = []
        s = self._scheduler(seen)
        running = _RunningBatch(num_reqs=3)
        decoding_batch = DecodingBatch(running)

        # Step N: decode is converted to extend because a peer rank extends.
        plan = s.get_next_batch_to_run(decoding_batch=decoding_batch, last_batch=None)
        self.assertIsNot(plan.batch_to_run, running)
        self.assertIs(plan.batch_to_run.forward_mode, _EXTEND)
        self.assertIs(plan.running_batch, running)
        self.assertIs(running.forward_mode, _DECODE)

        # Step N+1: one row finished; the converted step comes back as
        # last_batch and is not joined: its rows are the running rows.
        running.reqs[0].finished = True
        plan = s.get_next_batch_to_run(
            decoding_batch=decoding_batch, last_batch=plan.batch_to_run
        )

        self.assertEqual(len(seen), 2)
        self.assertIs(seen[-1], running)
        self.assertEqual(running.batch_size(), 2)
        self.assertIs(running.forward_mode, _DECODE)
        self.assertEqual(running.num_decode_rows, 0)


if __name__ == "__main__":
    unittest.main()
