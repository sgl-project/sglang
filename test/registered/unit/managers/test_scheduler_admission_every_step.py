"""Admission is attempted on every step, from the state of that step."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.min_free_slots_delayer import MinFreeSlotsDelayer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, AdmissionSkipReason
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _AdmissionAttempted(Exception):
    pass


class TestAdmissionSkipReason(CustomTestCase):
    def _scheduler(self, *, waiting: int, free_slots: int) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = [SimpleNamespace(rid=f"req-{i}") for i in range(waiting)]
        s.chunked_req = None
        s.min_free_slots_delayer = None
        s.get_num_allocatable_reqs = Mock(return_value=free_slots)
        s.processed_tokens_counter = 0
        # Reaching the policy means the skip checks were passed.
        s.policy = SimpleNamespace(calc_priority=Mock(side_effect=_AdmissionAttempted))
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


class _ScriptedAdder:
    """Answers add_one_req from the test's script; counts as admitted on CONTINUE."""

    results = []
    tried = []
    preempted = []

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = None

    def add_chunked_req(self, req):
        self.can_run_list.append(req)
        return req

    def add_one_req(self, req, **kwargs):
        self.tried.append(req)
        res = self.results.pop(0) if self.results else AddReqResult.CONTINUE
        if res == AddReqResult.CONTINUE:
            self.can_run_list.append(req)
        return res

    def preempt_to_schedule(self, req):
        self.preempted.append(req)
        return True


class TestAdmissionEveryStep(CustomTestCase):
    def setUp(self):
        _ScriptedAdder.results = []
        _ScriptedAdder.tried = []
        _ScriptedAdder.preempted = []

    def _scheduler(self):
        s = MagicMock()
        s.admission_skip_reason = MethodType(Scheduler.admission_skip_reason, s)
        s.grammar_manager.has_waiting_grammars.return_value = False
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.min_free_slots_delayer = None
        s.dynamic_chunk_sizer = None
        s.chunked_req = None
        s.enable_lora = False
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.enable_hierarchical_cache = False
        s.enable_unified_cache_external_linker = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.get_num_allocatable_reqs.return_value = 8
        s.is_mixed_chunk = False
        s.tp_worker.model_runner.prefill_aware_swa = False
        return s

    def _waiting(self, n, **kwargs):
        return [
            MagicMock(token_indices_to_pool=None, beam_group=None, **kwargs)
            for _ in range(n)
        ]

    def _admit(self, s, running_batch):
        with (
            patch.object(scheduler_module, "PrefillAdder", _ScriptedAdder),
            patch.object(scheduler_module, "PrefillStats"),
            patch.object(scheduler_module, "get_schedule"),
            patch.object(scheduler_module, "set_time_batch"),
            patch.object(
                scheduler_module.ScheduleBatch,
                "init_new",
                side_effect=lambda reqs, *args, **kwargs: MagicMock(reqs=reqs),
            ),
        ):
            ret, _ = Scheduler._get_new_batch_prefill_raw(
                s, prefill_delayer_single_pass=None, running_batch=running_batch
            )
        return ret

    def test_admits_on_the_step_after_no_token(self):
        # No running row leaves between the two passes; KV freed (or the
        # reserved ratio decayed) in between, so the head now fits.
        for running_reqs in (0, 1):
            with self.subTest(running_reqs=running_reqs):
                self.setUp()
                s = self._scheduler()
                head = self._waiting(1)[0]
                s.waiting_queue = [head]
                running_batch = ScheduleBatch(
                    reqs=[MagicMock() for _ in range(running_reqs)]
                )

                _ScriptedAdder.results = [AddReqResult.NO_TOKEN]
                self.assertIsNone(self._admit(s, running_batch))
                self.assertEqual(s.waiting_queue, [head])

                ret = self._admit(s, running_batch)
                self.assertEqual(s.policy.calc_priority.call_count, 2)
                self.assertEqual(ret.reqs, [head])
                self.assertEqual(s.waiting_queue, [])

    def test_chunked_continuation_tries_waiting_after_no_token(self):
        s = self._scheduler()
        head = self._waiting(1)[0]
        s.waiting_queue = [head]
        running_batch = ScheduleBatch(reqs=[MagicMock()])

        _ScriptedAdder.results = [AddReqResult.NO_TOKEN]
        self.assertIsNone(self._admit(s, running_batch))

        chunked_req = MagicMock(inflight_middle_chunks=0, token_indices_to_pool=None)
        s.chunked_req = chunked_req
        ret = self._admit(s, running_batch)
        self.assertEqual(_ScriptedAdder.tried, [head, head])
        self.assertEqual(ret.reqs, [chunked_req, head])

    def test_rows_full_stays_set_within_a_pass(self):
        # Only the beam candidate overflows the rows; once it did, the next
        # candidate of the same pass also needs a preemption.
        s = self._scheduler()
        s.enable_priority_preemption = True
        s.get_num_allocatable_reqs.side_effect = (
            lambda running_bs, beam_width=None, running_batch=None: (
                1 if beam_width is not None else 8
            )
        )
        first, last = self._waiting(2)
        beam = MagicMock(
            token_indices_to_pool=None, beam_group=SimpleNamespace(beam_width=2)
        )
        s.waiting_queue = [first, beam, last]

        self._admit(s, ScheduleBatch(reqs=[]))
        self.assertEqual(_ScriptedAdder.preempted, [beam, last])

        # A new pass starts with free rows again.
        self.setUp()
        s.waiting_queue = [first]
        self._admit(s, ScheduleBatch(reqs=[]))
        self.assertEqual(_ScriptedAdder.preempted, [])


if __name__ == "__main__":
    unittest.main()
