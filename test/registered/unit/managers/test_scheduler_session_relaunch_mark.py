"""The overlap loop marks a request whose result it processes when a launched or
planned batch already runs the request again; a streaming session then releases
the turn's recurrent state instead of keeping state the rerun advances."""

import unittest
from array import array
from unittest.mock import MagicMock

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req() -> Req:
    sampling_params = SamplingParams(max_new_tokens=8)
    sampling_params.normalize(None)
    return Req(
        rid="r",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3]),
        sampling_params=sampling_params,
        vocab_size=128,
    )


def _marks_at_processing(*, replan=True, sync=False, hybrid=True, sessions=True):
    """Run two overlap-loop iterations of one request and return the request's
    mark each time a result of it is processed."""
    req = _make_req()
    marks = []

    def make_batch(reqs):
        return ScheduleBatch(
            reqs=reqs,
            forward_mode=ForwardMode.DECODE,
            spec_algorithm=SpeculativeAlgorithm.NGRAM,
        )

    plans = iter([make_batch([req]), make_batch([req]) if replan else None])
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.gracefully_exit = False
    scheduler.ingest_requests = MagicMock(side_effect=[[], [], StopIteration])
    scheduler._engine_paused = False
    scheduler.enable_unified_memory = False
    scheduler.is_generation = False
    scheduler.is_hybrid_ssm = hybrid
    scheduler.running_batch = make_batch([req])
    scheduler.last_batch = None
    scheduler._apply_war_barrier = MagicMock()
    scheduler.on_idle = MagicMock()
    scheduler.get_next_batch_to_run = lambda *, running_batch, last_batch: MagicMock(
        running_batch=running_batch, batch_to_run=next(plans)
    )
    scheduler.is_disable_overlap_for_batch = lambda batch, last_batch: (
        sync and last_batch is not None
    )
    scheduler.run_batch = lambda batch: object()
    scheduler.process_batch_result = lambda batch, result: marks.append(
        req.relaunched_in_flight
    )

    with get_context().override_server_args(enable_streaming_session=sessions):
        with unittest.TestCase().assertRaises(StopIteration):
            scheduler.event_loop_overlap()
    return marks


class TestSessionRelaunchMark(unittest.TestCase):
    def test_request_rerun_before_its_result_is_marked(self):
        # Overlap: the second batch is launched before the first result.
        self.assertEqual(_marks_at_processing(), [True])
        # Synchronous step: the second batch is planned before the first result.
        self.assertEqual(_marks_at_processing(sync=True), [True])

    def test_request_not_rerun_is_not_marked(self):
        self.assertEqual(_marks_at_processing(replan=False), [False])

    def test_marking_needs_recurrent_state_and_streaming_sessions(self):
        for case in (dict(hybrid=False), dict(sessions=False)):
            with self.subTest(**case):
                self.assertEqual(_marks_at_processing(**case), [False])


if __name__ == "__main__":
    unittest.main()
