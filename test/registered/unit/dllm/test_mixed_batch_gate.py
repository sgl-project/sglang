"""Unit tests for the dLLM mixed-round gate.

``SGLANG_ENABLE_DLLM_MIXED_BATCH`` lets one round hold both prefill-phase and
decode-phase rows. The mixed path is a restriction of the either/or path rather
than a replacement: it walks the waiting queue in arrival order instead of
prefill-first, and it does not wire priority preemption. Those costs are only
worth paying when the round actually has decode work that would otherwise wait
behind a partially filled prefill round, so the gate has to keep single-phase
and already-full rounds on the original path.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.schedule_policy import AddReqResult

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestShouldMixDllmBatches(CustomTestCase):
    def _gate(self, prefill: int, decode: int, capacity: int, preempt: bool = False):
        return SchedulerDllmMixin._should_mix_dllm_batches(
            num_prefill_reqs=prefill,
            num_decode_reqs=decode,
            round_capacity=capacity,
            priority_preemption_enabled=preempt,
        )

    def test_mixes_when_prefill_leaves_room_for_decode(self):
        # The case the feature exists for: prefill alone would leave the round
        # partly empty while decode rows wait.
        self.assertTrue(self._gate(prefill=2, decode=5, capacity=8))

    def test_declines_when_prefill_already_fills_the_round(self):
        # Nothing to steal into: mixing would only cost the prefill-first
        # ordering and preemption the either/or path keeps.
        self.assertFalse(self._gate(prefill=8, decode=5, capacity=8))
        self.assertFalse(self._gate(prefill=9, decode=5, capacity=8))

    def test_declines_on_a_single_phase_round(self):
        # A round with only one phase cannot be mixed by definition.
        self.assertFalse(self._gate(prefill=4, decode=0, capacity=8))
        self.assertFalse(self._gate(prefill=0, decode=4, capacity=8))

    def test_declines_when_the_round_has_no_capacity(self):
        # running_batch already holds max_running_reqs rows.
        self.assertFalse(self._gate(prefill=2, decode=5, capacity=0))

    def test_declines_when_priority_preemption_is_enabled(self):
        # The mixed path does not attempt preemption when the round fills, so
        # taking it under --enable-priority-scheduling would silently drop a
        # feature the either/or path provides. Same inputs as the mixing case
        # above, so only the preemption flag decides.
        self.assertTrue(self._gate(prefill=2, decode=5, capacity=8, preempt=False))
        self.assertFalse(self._gate(prefill=2, decode=5, capacity=8, preempt=True))


class _PhaseReq:
    def __init__(self, phase):
        self.dllm_phase = phase

    def is_dllm_prefill(self):
        return self.dllm_phase in (
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        )


class TestMixedRoundCounter(CustomTestCase):
    """``dllm_num_mixed_rounds`` counts rounds the MIXED COMPOSER built with
    both phases admitted — at composition, not by re-reading phases when the
    round is recorded. ``determine_dllm_phase`` can flip an incoming prefill
    row to STAGING_DECODE during an either/or round's admission (a cached
    prefix covers its first blocks), so counting phases in
    ``_update_state_for_batch`` let a flags-off run report a mixed round.
    The flags-off counter must be structurally zero: only the composer
    increments it."""

    @staticmethod
    def _compose(admitted, *, staging_result=AddReqResult.CONTINUE):
        """Run the composer unbound on a stub whose admission passes admit
        ``admitted`` (phase-flips included: the rows land with their final
        phases, as process_dllm_incoming_reqs leaves them)."""
        adder = SimpleNamespace(can_run_list=[])

        def staging_pass(a, reqs):
            a.can_run_list.extend(r for r in reqs if r in admitted)
            return staging_result

        def incoming_pass(a, reqs, *, running_batch):
            a.can_run_list.extend(r for r in reqs if r in admitted)
            return AddReqResult.CONTINUE

        scheduler = SimpleNamespace(
            dllm_manager=SimpleNamespace(waiting_queue=list(admitted)),
            process_dllm_staging_reqs=staging_pass,
            process_dllm_incoming_reqs=incoming_pass,
            dllm_num_mixed_rounds=0,
        )
        SchedulerDllmMixin._process_dllm_batches_mixed(
            scheduler, adder, running_batch=SimpleNamespace(reqs=[])
        )
        return scheduler

    def test_composer_counts_a_round_with_both_phases(self):
        scheduler = self._compose(
            [
                _PhaseReq(DllmReqPhase.STAGING_DECODE),
                _PhaseReq(DllmReqPhase.STAGING_PREFILL),
            ]
        )
        self.assertEqual(scheduler.dllm_num_mixed_rounds, 1)

    def test_composer_skips_a_single_phase_round(self):
        # The composer was asked to mix but every admitted row ended up
        # decode-phase (e.g. the prefill row phase-flipped on a cache hit):
        # the round that runs is single-phase, so it does not count.
        scheduler = self._compose(
            [
                _PhaseReq(DllmReqPhase.STAGING_DECODE),
                _PhaseReq(DllmReqPhase.STAGING_DECODE),
            ]
        )
        self.assertEqual(scheduler.dllm_num_mixed_rounds, 0)

    def test_composer_skips_an_empty_round(self):
        scheduler = self._compose([])
        self.assertEqual(scheduler.dllm_num_mixed_rounds, 0)

    def test_refused_staging_round_still_counts_what_it_admitted(self):
        # A per-row staging refusal (NO_TOKEN) skips the incoming pass but the
        # admitted staging rows still run; two-phased, the round counts.
        scheduler = self._compose(
            [
                _PhaseReq(DllmReqPhase.STAGING_DECODE),
                _PhaseReq(DllmReqPhase.STAGING_PREFILL),
            ],
            staging_result=AddReqResult.NO_TOKEN,
        )
        self.assertEqual(scheduler.dllm_num_mixed_rounds, 1)

    def test_recording_a_round_does_not_count_it(self):
        # The either/or path records rounds through _update_state_for_batch
        # only; a two-phased can_run_list there (the cache-hit phase flip)
        # must not touch the mixed counter.
        scheduler = SimpleNamespace(
            dllm_num_rounds=0,
            dllm_num_mixed_rounds=0,
        )
        can_run_list = [
            _PhaseReq(DllmReqPhase.STAGING_PREFILL),
            _PhaseReq(DllmReqPhase.STAGING_DECODE),
        ]
        SchedulerDllmMixin._update_state_for_batch(
            scheduler, can_run_list, SimpleNamespace(preempt_list=[])
        )
        self.assertEqual(scheduler.dllm_num_rounds, 1)
        self.assertEqual(scheduler.dllm_num_mixed_rounds, 0)


if __name__ == "__main__":
    unittest.main()
