"""A mixed step that carries a grammar resolves the pending result before it runs.

Speculative algorithms with grammar overlap advance the grammar FSM inside
verify() (the scheduler's grammar barrier). A MIXED step runs no verify(): its
running requests take a plain one-token decode whose bitmask comes from the
FSM, so the previous batch's result must be processed first, or a JSON request
that just emitted '{' is masked as if it had not and may emit '{"' next.
"""

import unittest
from collections import deque
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _batch(mode, algorithm, has_grammar):
    return SimpleNamespace(
        forward_mode=mode,
        spec_algorithm=algorithm,
        has_grammar=has_grammar,
        is_extend_in_batch=mode.is_extend(),
        grammar_needs_sync=lambda: (
            has_grammar and not algorithm.supports_grammar_overlap()
        ),
    )


class TestMixedStepGrammarSync(CustomTestCase):
    def _disables_overlap(
        self,
        mode,
        algorithm=SpeculativeAlgorithm.EAGLE,
        has_grammar=True,
        pending_results=1,
    ):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.require_mlp_sync = False
        scheduler.result_queue = deque([object()] * pending_results)
        return scheduler.is_disable_overlap_for_batch(
            _batch(mode, algorithm, has_grammar),
            _batch(ForwardMode.DECODE, algorithm, has_grammar),
        )

    def test_mixed_step_with_grammar_syncs(self):
        self.assertTrue(self._disables_overlap(ForwardMode.MIXED))

    def test_mixed_step_without_grammar_overlaps(self):
        self.assertFalse(self._disables_overlap(ForwardMode.MIXED, has_grammar=False))

    def test_mixed_step_with_nothing_pending_overlaps(self):
        self.assertFalse(self._disables_overlap(ForwardMode.MIXED, pending_results=0))

    def test_mixed_step_without_speculation_overlaps(self):
        # Non-speculative batches delay their sampling until the result is in.
        self.assertFalse(
            self._disables_overlap(
                ForwardMode.MIXED, algorithm=SpeculativeAlgorithm.NONE
            )
        )

    def test_decode_step_keeps_grammar_overlap(self):
        # The grammar barrier inside verify() covers decode steps.
        self.assertFalse(self._disables_overlap(ForwardMode.DECODE))


if __name__ == "__main__":
    unittest.main()
