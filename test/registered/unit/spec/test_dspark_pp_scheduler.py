"""Request relay and pause ordering at the synchronous pipeline entry point."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSparkPipelineLoop(CustomTestCase):
    def exercise(
        self, *, last_stage=False, pause=False, pd_mode=DisaggregationMode.NULL
    ):
        events = []
        batch = SimpleNamespace(reqs=[object()])
        scheduler = SimpleNamespace(
            gracefully_exit=False,
            _engine_paused=False,
            disaggregation_mode=pd_mode,
            server_args=SimpleNamespace(),
            waiting_queue=[],
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
        scheduler.get_next_disagg_prefill_batch_to_run = plan
        scheduler.get_next_disagg_decode_batch_to_run = plan
        scheduler.process_decode_queue = lambda: events.append(("decode_queue",))
        scheduler.process_disagg_prefill_inflight_queue = lambda: events.append(
            ("inflight",)
        )
        scheduler._pp_dspark_prefill_handoffs = lambda current, result: events.append(
            ("handoff",)
        )

        def bootstrap():
            events.append(("bootstrap",))
            return []

        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            pop_bootstrapped=bootstrap
        )
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
            expected.append(("process", request))
            if pd_mode == DisaggregationMode.PREFILL:
                expected.append(("bootstrap",))
            elif pd_mode == DisaggregationMode.DECODE:
                expected.append(("decode_queue",))
            expected.append(("plan",))
            expected.extend(
                [("coordinate_idle", None), ("idle",)]
                if request == "idle"
                else [("run",)]
                + ([("handoff",)] if pd_mode == DisaggregationMode.PREFILL else [])
                + [("result",)]
            )
            if pd_mode == DisaggregationMode.PREFILL:
                expected.append(("inflight",))
        self.assertEqual(events, expected)
        self.assertIsNone(scheduler.last_batch)
        self.assertIsNone(scheduler.mbs[0])

    def test_requests_relay_before_processing_and_batch_execution(self):
        self.exercise()

    def test_last_stage_does_not_relay_back_to_first(self):
        self.exercise(last_stage=True)

    def test_pause_keeps_receiving_and_resume_preserves_batch_state(self):
        self.exercise(pause=True)

    def test_prefill_coordinates_bootstrap_handoff_and_release(self):
        self.exercise(pd_mode=DisaggregationMode.PREFILL)

    def test_decode_advances_queues_before_batch_planning(self):
        self.exercise(pd_mode=DisaggregationMode.DECODE)

    def test_teacher_copy_completes_before_broadcast_and_accept(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                events = []
                capture = SimpleNamespace(
                    pack_pp_handoffs=lambda batch, tokens, events=events: events.append(
                        "pack"
                    )
                    or (b"teacher",),
                    accept_pp_handoffs=lambda batch, payload, events=events: events.append(
                        ("accept", payload)
                    ),
                )
                scheduler = SimpleNamespace(
                    tp_worker=SimpleNamespace(training_capture=capture),
                    pp_group=SimpleNamespace(world_size=2),
                    tp_group=SimpleNamespace(world_size=1),
                    world_group=SimpleNamespace(
                        rank_in_group=rank,
                        broadcast_object=lambda payload, src, events=events: events.append(
                            ("broadcast", src)
                        )
                        or (b"teacher",),
                    ),
                )
                result = SimpleNamespace(
                    copy_done=SimpleNamespace(
                        synchronize=lambda events=events: events.append("copy")
                    ),
                    next_token_ids=object(),
                )
                SchedulerPPMixin._pp_dspark_prefill_handoffs(
                    scheduler, object(), result
                )
                self.assertEqual(
                    events,
                    (["copy", "pack"] if rank == 1 else [])
                    + [("broadcast", 1), ("accept", (b"teacher",))],
                )


if __name__ == "__main__":
    unittest.main()
