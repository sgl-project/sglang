"""An empty running batch never keeps admission closed (#41841)."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _AdmissionAttempted(Exception):
    pass


class TestEmptyBatchFullLatch(CustomTestCase):
    def _scheduler(self) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = [SimpleNamespace(rid="req-0")]
        s.chunked_req = None
        s.min_free_slots_delayer = None
        s.get_num_allocatable_reqs = Mock(return_value=8)
        s.processed_tokens_counter = 0
        # Reaching the policy means the admission gates were passed.
        s.policy = SimpleNamespace(calc_priority=Mock(side_effect=_AdmissionAttempted))
        return s

    def test_latched_empty_batch_still_admits(self):
        # Whichever pass set the flag on an empty batch (no KV room, or no free
        # request slot), no later step shrinks the batch to clear it.
        s = self._scheduler()
        running_batch = ScheduleBatch(reqs=[], batch_is_full=True)

        with self.assertRaises(_AdmissionAttempted):
            s._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None, running_batch=running_batch
            )
        self.assertFalse(running_batch.batch_is_full)


if __name__ == "__main__":
    unittest.main()
