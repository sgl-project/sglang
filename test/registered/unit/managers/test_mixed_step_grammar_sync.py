"""A mixed step's grammar bitmap must see the previous batch's result.

Spec workers with grammar overlap advance the FSM inside verify() (the
grammar barrier), so decode steps stay overlapped. A MIXED batch is
forward_mode.is_extend(), so a spec worker runs it as a plain prefill
(_forward_prefill_batch) whose grammar mask is built from req.grammar with no
verify() barrier. With an unprocessed result from the previous batch still in
the queue, that mask reflects the FSM before the previous decode step's
tokens, and the mixed batch's decode rows can sample a token accept_token then
rejects (finish_reason abort). The previous result must be processed before
the mixed batch launches.

Note: in overlap spec the scheduler's running_batch is an empty shell
(decode state lives inside the spec worker), so the decision must gate on the
pending result itself, not on running_batch being non-empty.
"""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import PropertyMock

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _batch(mode, algorithm, has_grammar):
    needs_sync = (
        has_grammar
        and not algorithm.is_none()
        and not algorithm.supports_grammar_overlap()
    )
    return SimpleNamespace(
        forward_mode=mode,
        spec_algorithm=algorithm,
        has_grammar=has_grammar,
        is_extend_in_batch=mode.is_extend(),
        grammar_needs_sync=lambda: needs_sync,
    )


class TestMixedStepGrammarSync(CustomTestCase):
    def _early(self, last_batch, running=1, waiting=0, chunked=None, pending=1):
        s = Scheduler.__new__(Scheduler)
        s.require_mlp_sync = False
        s.result_queue = deque([object()] * pending)
        s.running_batch = (
            SimpleNamespace(reqs=[object()] * running) if running else None
        )
        s.waiting_queue = [object()] * waiting
        with unittest.mock.patch.object(
            Scheduler, "chunked_req", new_callable=PropertyMock, return_value=chunked
        ):
            return s.should_process_last_batch_early(last_batch)

    def _spec(self, mode, has_grammar=True):
        return _batch(mode, SpeculativeAlgorithm.EAGLE, has_grammar)

    def test_mixed_grammar_processes_early(self):
        # Prefill work pending while a queued grammar decode result exists:
        # the next step is mixed.
        last = self._spec(ForwardMode.DECODE)
        self.assertTrue(self._early(last, running=2, waiting=3))

    def test_mixed_from_chunked_processes_early(self):
        last = self._spec(ForwardMode.DECODE)
        self.assertTrue(self._early(last, running=1, waiting=0, chunked=object()))

    def test_overlap_spec_empty_running_shell_still_processes(self):
        # In overlap spec the scheduler's running_batch is an empty shell
        # (decode state is inside the spec worker); a pending result IS the
        # in-flight-decode signal, so the decision must still fire.
        last = self._spec(ForwardMode.DECODE)
        self.assertTrue(self._early(last, running=0, waiting=2))

    def test_pure_decode_step_keeps_overlap(self):
        # No prefill work: the next step is a decode whose verify() barrier
        # advances the FSM over the queued result.
        last = self._spec(ForwardMode.DECODE)
        self.assertFalse(self._early(last, running=2, waiting=0))

    def test_last_batch_without_grammar_does_not(self):
        last = self._spec(ForwardMode.DECODE, has_grammar=False)
        self.assertFalse(self._early(last, running=2, waiting=3))

    def test_non_spec_batch_does_not(self):
        # Non-spec batches delay their sampling until the result is in.
        last = _batch(ForwardMode.DECODE, SpeculativeAlgorithm.NONE, True)
        self.assertFalse(self._early(last, running=2, waiting=3))

    def test_nothing_pending_does_not(self):
        last = self._spec(ForwardMode.DECODE)
        self.assertFalse(self._early(last, running=2, waiting=3, pending=0))

    def test_no_prefill_work_does_not(self):
        # Nothing queued and no chunk: a pure decode step follows, and its
        # verify() barrier resolves the FSM.
        last = self._spec(ForwardMode.DECODE)
        self.assertFalse(self._early(last, running=2, waiting=0))

    def test_last_batch_none(self):
        self.assertFalse(self._early(None, running=2, waiting=3))


if __name__ == "__main__":
    unittest.main()
