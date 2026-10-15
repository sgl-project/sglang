"""Under overlap, a spec step with grammar requests launches before the previous
step's host commit; that step's grammar commit runs inside the launch."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
    advance_grammar_fsm,
    commit_pending_grammar,
)
from sglang.srt.managers.scheduler_components.engine_state import (  # noqa: E402
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.mode_owners import (  # noqa: E402
    ModeOwners,
)
from sglang.srt.managers.scheduler_components.mode_stages import (  # noqa: E402
    ScheduleSlot,
)
from sglang.srt.managers.scheduler_components.pending_steps import (  # noqa: E402
    PendingSteps,
)
from sglang.srt.managers.utils import GenerationBatchResult  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Grammar:
    def __init__(self, events):
        self.events = events

    def accept_token(self, token_id):
        self.events.append(("accept", token_id))

    def is_terminated(self):
        return False


class _Req:
    def __init__(self, grammar):
        self.rid = "r0"
        self.grammar = grammar
        self.is_retracted = False
        self.lora_id = None

    def finished(self):
        return False


def _spec_decode_step(req, forward_iter):
    step = ScheduleBatch(reqs=[req])
    step.forward_mode = ForwardMode.DECODE
    step.spec_algorithm = SpeculativeAlgorithm.EAGLE
    step.has_grammar = True
    step.forward_iter = forward_iter
    return step


def _spec_result(tokens):
    return GenerationBatchResult(
        next_token_ids=torch.tensor(tokens),
        accept_lens=torch.tensor([len(tokens)]),
        speculative_num_draft_tokens=len(tokens),
    )


class TestGrammarCommitOverlap(CustomTestCase):
    def test_spec_grammar_step_launches_before_the_previous_commit(self):
        events = []
        req = _Req(_Grammar(events))
        steps = [_spec_decode_step(req, 1), _spec_decode_step(req, 2)]
        results = {1: _spec_result([5, 6]), 2: _spec_result([7, 8])}
        processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)

        s = Scheduler.__new__(Scheduler)
        s.mode_owners = ModeOwners()
        s.counters = LaunchCounters()
        s.flags = EngineFlags()
        s.flags.gracefully_exit = False
        s.flat_slot = ScheduleSlot()
        s.disaggregation_mode = DisaggregationMode.NULL
        s.mode_owners.disagg_prefill = None
        s.mode_owners.disagg_decode = None
        s.mode_owners.dllm = None
        s.mode_owners.hisparse_coordinator = None
        s.ingest_requests = MagicMock(side_effect=[[], [], StopIteration])
        s.flags._engine_paused = False
        s._sched_idled = False
        s.pending = PendingSteps()
        s.require_mlp_sync = False
        s.spec_algorithm = SpeculativeAlgorithm.EAGLE
        s.recv_skipper = None
        s.is_generation = False
        s.batch_result_processor = processor
        s.get_next_batch_to_run = MagicMock(side_effect=steps)
        s._apply_war_barrier = MagicMock()
        s.step_counters = MagicMock()

        def run_batch(step):
            events.append(("launch", step.forward_iter))
            # The worker's verify runs the pending grammar commit before it
            # builds this step's vocab mask.
            commit_pending_grammar(s.pending)
            return results[step.forward_iter]

        def process_batch_result(step, result):
            events.append(("commit", step.forward_iter))
            # The host commit runs the step's grammar commit if not done yet.
            advance_grammar_fsm(result, step)

        s.step_launcher = SimpleNamespace(run_batch=run_batch)
        s.process_batch_result = process_batch_result
        s.init_mode_stages()

        with self.assertRaises(StopIteration):
            s.event_loop_overlap()

        self.assertEqual(
            events,
            [
                ("launch", 1),
                ("launch", 2),
                ("accept", 5),
                ("accept", 6),
                ("commit", 1),
            ],
        )
        self.assertEqual(results[1].grammar_retained_tokens, [[5, 6]])


if __name__ == "__main__":
    unittest.main()
