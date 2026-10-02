"""Request relay and pause ordering at the synchronous pipeline entry point."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSparkPipelineLoop(CustomTestCase):
    def exercise(self, *, last_stage=False, pause=False):
        events = []
        batch = SimpleNamespace(reqs=[object()])
        scheduler = SimpleNamespace(
            gracefully_exit=False,
            _engine_paused=False,
            model_worker=object(),
            world_group=object(),
            pp_group=SimpleNamespace(is_last_rank=last_stage),
            tp_group=object(),
            running_batch=object(),
            last_batch=None,
            running_mbs=[None],
            mbs=[None],
            init_pp_loop_state=Mock(),
        )
        turns = iter(["pause", "resume", "idle"] if pause else ["run", "idle"])

        def receive():
            request = next(turns)
            events.append(("receive", request))
            return request

        def process(request):
            events.append(("process", request))
            scheduler._engine_paused = request == "pause"

        def plan(**kwargs):
            self.assertIs(kwargs["running_batch"], scheduler.running_batch)
            events.append(("plan",))
            return SimpleNamespace(
                running_batch=scheduler.running_batch,
                batch_to_run=None if scheduler.last_batch else batch,
            )

        def run(current):
            self.assertIs(scheduler.mbs[0], current)
            self.assertIs(scheduler.running_mbs[0], scheduler.running_batch)
            events.append(("run",))
            return object()

        def idle():
            events.append(("idle",))
            scheduler.gracefully_exit = True

        scheduler.request_receiver = SimpleNamespace(recv_requests=receive)
        scheduler.process_input_requests = process
        scheduler._pp_send_pyobj_to_next_stage = lambda request: events.append(
            ("relay", request)
        )
        scheduler.get_next_batch_to_run = plan
        scheduler.run_batch = run
        scheduler.process_batch_result = lambda current, result: events.append(
            ("result",)
        )
        scheduler.on_idle = idle
        coordinator = Mock()
        coordinator.run_batch.side_effect = lambda current: events.append(
            ("coordinate_idle", current)
        )
        with (
            patch(
                "sglang.srt.speculative.dspark_components.dspark_pp_coordinator."
                "DSparkPPCoordinator",
                return_value=coordinator,
            ),
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(False),
        ):
            SchedulerPPMixin.event_loop_pp_dspark(scheduler)
        prefix = []
        if pause:
            prefix = [("receive", "pause")]
            if not last_stage:
                prefix.append(("relay", "pause"))
            prefix.append(("process", "pause"))
        expected = prefix
        for request in ("resume" if pause else "run", "idle"):
            expected.append(("receive", request))
            if not last_stage:
                expected.append(("relay", request))
            expected.extend([("process", request), ("plan",)])
            expected.extend(
                [("coordinate_idle", None), ("idle",)]
                if request == "idle"
                else [("run",), ("result",)]
            )
        self.assertEqual(events, expected)
        self.assertIsNone(scheduler.last_batch)
        self.assertIsNone(scheduler.mbs[0])

    def test_requests_relay_before_processing_and_batch_execution(self):
        self.exercise()

    def test_last_stage_does_not_relay_back_to_first(self):
        self.exercise(last_stage=True)

    def test_pause_keeps_receiving_and_resume_preserves_batch_state(self):
        self.exercise(pause=True)


if __name__ == "__main__":
    unittest.main()
