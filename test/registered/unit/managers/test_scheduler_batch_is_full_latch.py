"""An empty decoding list never keeps admission closed (#41841)."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.engine_state import (
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.pending_steps import (
    PendingStep,
    PendingSteps,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _AdmissionAttempted(Exception):
    pass


class _Req(SimpleNamespace):
    # Hashed by identity, as Req is.
    __hash__ = object.__hash__


class TestEmptyBatchFullLatch(CustomTestCase):
    def _scheduler(self) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.counters = LaunchCounters()
        s.flags = EngineFlags()
        s.lists = CoreLists()
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.lists.waiting_queue = [SimpleNamespace(rid="req-0")]
        s.lists.decoding = []
        s.lists.chunked_req = None
        s.min_free_slots_delayer = None
        s.beam_coordinator = SimpleNamespace(
            get_num_allocatable_reqs=Mock(return_value=8)
        )
        s.counters.processed_tokens_counter = 0
        # Reaching the policy means the admission gates were passed.
        s.policy = SimpleNamespace(calc_priority=Mock(side_effect=_AdmissionAttempted))
        return s

    def test_latched_flag_with_nothing_decoding_still_admits(self):
        # Whichever pass set the flag with nothing decoding (no KV room, or no
        # free request slot), no later commit removes a request to clear it.
        s = self._scheduler()
        slot = ScheduleSlot(batch_is_full=True)

        with self.assertRaises(_AdmissionAttempted):
            s._get_new_batch_prefill_raw(slot, prefill_delayer_single_pass=None)
        self.assertFalse(slot.batch_is_full)

    def test_skipping_a_request_finishing_in_flight_clears_the_flag(self):
        # The decode step and a mixed step's decode rows both select here.
        def req(rid, max_new_tokens):
            return _Req(
                rid=rid,
                is_prefill_only=False,
                beam_group=None,
                output_ids=[1],
                sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
            )

        finishing, running = req("a", 2), req("b", 8)
        s = self._scheduler()
        s.lists.decoding = [finishing, running]
        s.flags.enable_skip_finishing_decode = True
        # One decode step of both is in flight.
        s.pending = PendingSteps()
        step = SimpleNamespace(
            reqs=[finishing, running], forward_mode=ForwardMode.DECODE, chunked_req=None
        )
        slot = ScheduleSlot(batch_is_full=True)
        s.pending.push(
            PendingStep(
                step=step, result=None, slot=slot, launch_ts=0.0, after_idle_gap=False
            )
        )

        self.assertEqual(s._select_decode_reqs(slot), [running])
        self.assertFalse(slot.batch_is_full)


if __name__ == "__main__":
    unittest.main()
