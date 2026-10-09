"""A running batch rebuilt from a DP-converted extend batch reopens admission."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import NextBatchPlan, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ConvertedBatch:
    """The old running batch after DP attention converted it to extend: it is
    the same object and still carries the flag it had before the step."""

    def __init__(self, num_reqs, num_finished):
        self.reqs = [
            SimpleNamespace(finished=i < num_finished) for i in range(num_reqs)
        ]
        self.forward_mode = SimpleNamespace(is_extend=lambda: True)
        self.chunked_req = None
        self.is_prefill_only = False
        self.batch_is_full = True

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, chunked_req_to_exclude=()):
        self.reqs = [
            r for r in self.reqs if not r.finished and r not in chunked_req_to_exclude
        ]


class TestDpConvertedBatchFull(unittest.TestCase):
    def test_finished_rows_reopen_admission(self):
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

        seen = []

        def get_new_batch_prefill(running_batch):
            seen.append((running_batch, running_batch.batch_is_full))
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

        s.get_new_batch_prefill = get_new_batch_prefill

        # The empty placeholder left in place of the converted batch copies
        # its flag (see the end of get_next_batch_to_run).
        placeholder = ScheduleBatch(reqs=[], batch_is_full=True)
        # One of its three rows finished, so the step freed room.
        last_batch = _ConvertedBatch(num_reqs=3, num_finished=1)
        s.get_next_batch_to_run(running_batch=placeholder, last_batch=last_batch)

        ((running_batch, batch_is_full),) = seen
        self.assertIs(running_batch, last_batch)
        self.assertFalse(batch_is_full)


if __name__ == "__main__":
    unittest.main()
