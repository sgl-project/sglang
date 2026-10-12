"""An empty running batch never keeps admission closed (#41841)."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.min_free_slots_delayer import MinFreeSlotsDelayer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AdmissionSkipReason
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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


class TestAdmissionSkipReason(CustomTestCase):
    def _scheduler(self, *, waiting: int, free_slots: int) -> Scheduler:
        s = TestEmptyBatchFullLatch._scheduler(self)
        s.waiting_queue = [SimpleNamespace(rid=f"req-{i}") for i in range(waiting)]
        s.get_num_allocatable_reqs = Mock(return_value=free_slots)
        return s

    def _admit(self, s: Scheduler, running_reqs: int = 0) -> bool:
        """Run one admission pass; True if it got past the skip checks."""
        running_batch = ScheduleBatch(
            reqs=[SimpleNamespace(rid=f"running-{i}") for i in range(running_reqs)]
        )
        try:
            ret, _ = s._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None, running_batch=running_batch
            )
        except _AdmissionAttempted:
            return True
        self.assertIsNone(ret)
        return False

    def test_no_waiting_skips(self):
        s = self._scheduler(waiting=0, free_slots=8)
        self.assertIs(
            s.admission_skip_reason(ScheduleBatch(reqs=[])),
            AdmissionSkipReason.NO_WAITING,
        )
        self.assertFalse(self._admit(s))

    def test_no_req_slot_skips(self):
        s = self._scheduler(waiting=1, free_slots=0)
        self.assertIs(
            s.admission_skip_reason(ScheduleBatch(reqs=[])),
            AdmissionSkipReason.NO_REQ_SLOT,
        )
        self.assertFalse(self._admit(s))

    def test_priority_preemption_attempts_without_free_slots(self):
        s = self._scheduler(waiting=1, free_slots=0)
        s.enable_priority_preemption = True
        self.assertIsNone(s.admission_skip_reason(ScheduleBatch(reqs=[])))
        self.assertTrue(self._admit(s))

    def test_chunked_req_attempts_without_waiting_or_slots(self):
        s = self._scheduler(waiting=0, free_slots=0)
        s.chunked_req = SimpleNamespace(rid="chunked")
        self.assertIsNone(s.admission_skip_reason(ScheduleBatch(reqs=[])))
        self.assertTrue(self._admit(s))

    def test_min_free_slots_delayer_skips(self):
        # A policy deferral, not a skip reason: the step has a free slot.
        s = self._scheduler(waiting=1, free_slots=1)
        s.min_free_slots_delayer = MinFreeSlotsDelayer(min_free_slots=2)
        self.assertIsNone(s.admission_skip_reason(ScheduleBatch(reqs=[])))
        self.assertFalse(self._admit(s, running_reqs=1))


if __name__ == "__main__":
    unittest.main()
