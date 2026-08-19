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

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin

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


if __name__ == "__main__":
    unittest.main()
