"""The overlap loop must not launch a host-draft grammar batch with a request
that the previous result, processed at the grammar sync, already finished."""

import unittest
from array import array
from unittest.mock import MagicMock

from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req() -> Req:
    sampling_params = SamplingParams(max_new_tokens=8)
    sampling_params.normalize(None)
    req = Req(
        rid="r",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3]),
        sampling_params=sampling_params,
        vocab_size=128,
    )
    # Four tokens in: the next single token cannot finish it by length, so
    # only processing the pending result reveals that it finished.
    req.output_ids.extend([4, 5, 6, 7])
    return req


def _run_two_iterations(spec_algorithm, has_grammar, first_mode=ForwardMode.DECODE):
    """Drive event_loop_overlap through two plans of one request whose first
    result finishes it (a multi-token run reaching max_new_tokens)."""
    req = _make_req()
    events, launched = [], []

    def make_batch(reqs, mode):
        return ScheduleBatch(
            reqs=reqs,
            forward_mode=mode,
            spec_algorithm=spec_algorithm,
            has_grammar=has_grammar,
        )

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.gracefully_exit = False
    scheduler.ingest_requests = MagicMock(side_effect=[[], [], StopIteration])
    scheduler._engine_paused = False
    scheduler.require_mlp_sync = False
    scheduler.enable_unified_memory = False
    scheduler.is_generation = False
    scheduler.spec_algorithm = spec_algorithm
    # A prefill runs from an empty running batch and merges at the next plan.
    first_is_decode = first_mode.is_decode()
    scheduler.running_batch = ScheduleBatch(reqs=[req] if first_is_decode else [])
    scheduler.running_batch.has_grammar = has_grammar and first_is_decode
    scheduler.last_batch = None
    scheduler._apply_war_barrier = MagicMock()

    def get_next_batch_to_run(*, running_batch, last_batch):
        events.append("plan")
        if last_batch is None:
            return MagicMock(
                running_batch=running_batch, batch_to_run=make_batch([req], first_mode)
            )
        # Like the real planner: finished requests leave the running batch.
        live = [r for r in [req] if not r.finished()]
        batch = make_batch(live, ForwardMode.DECODE) if live else None
        return MagicMock(running_batch=running_batch, batch_to_run=batch)

    def run_batch(batch):
        events.append("run")
        launched.append([(r.rid, r.finished()) for r in batch.reqs])
        return object()

    def process_batch_result(batch, result):
        events.append("process")
        # The committed run finishes the request, which releases its KV row.
        req.output_ids.extend([8, 9, 10, 11])
        req.finished_reason = FINISH_LENGTH(length=8)

    scheduler.get_next_batch_to_run = get_next_batch_to_run
    scheduler.run_batch = run_batch
    scheduler.process_batch_result = process_batch_result

    with unittest.TestCase().assertRaises(StopIteration):
        scheduler.event_loop_overlap()
    return events, launched


class TestGrammarSyncBeforePlan(unittest.TestCase):
    def test_ngram_grammar_result_is_processed_before_next_plan(self):
        for first_mode in (ForwardMode.DECODE, ForwardMode.EXTEND):
            with self.subTest(first_mode=first_mode.name):
                events, launched = _run_two_iterations(
                    SpeculativeAlgorithm.NGRAM, True, first_mode
                )
                # The finished request is never launched again.
                self.assertEqual(launched, [[("r", False)]])
                self.assertEqual(events, ["plan", "run", "process", "plan"])

    def test_overlapping_algorithms_keep_plan_before_process(self):
        cases = (
            (SpeculativeAlgorithm.NGRAM, False),
            (SpeculativeAlgorithm.EAGLE, True),
            (SpeculativeAlgorithm.NONE, True),
        )
        for spec_algorithm, has_grammar in cases:
            with self.subTest(spec=spec_algorithm.name, grammar=has_grammar):
                events, launched = _run_two_iterations(spec_algorithm, has_grammar)
                # Ordinary overlap: the second batch launches before the first
                # result is processed, so the request is still live at launch.
                self.assertEqual(launched, [[("r", False)], [("r", False)]])
                self.assertEqual(events, ["plan", "run", "plan", "run", "process"])


if __name__ == "__main__":
    unittest.main()
