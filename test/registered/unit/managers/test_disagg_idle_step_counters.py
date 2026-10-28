"""Idle gaps must not inflate prefill busy time or decode timing samples."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from parameterized import parameterized

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_assembly import init_mode_stages  # noqa: E402
from sglang.srt.managers.scheduler_components.bundles import Streams
from sglang.srt.managers.scheduler_components.engine_state import (
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.pending_steps import (
    PendingStep,
    PendingSteps,
)
from sglang.srt.managers.scheduler_components.step_counters import StepCounters
from sglang.srt.managers.scheduler_components.step_launcher import StepLauncher
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import decode_owner_shell, prefill_owner_shell
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402
from sglang.test.scheduler_request_lists import make_request_retirer
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    published_topology,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

LAUNCH_TIMESTAMPS = (0.0, 0.125, 1.0, 1.125)
# Prefill busy time is charged launch -> result, so the result clock matters too.
RESULT_TIMESTAMPS = tuple(ts + 0.1 for ts in LAUNCH_TIMESTAMPS)
PP_MODULE = "sglang.srt.managers.scheduler_components.pp_ring"
PD_RING_MODULE = "sglang.srt.disaggregation.pd_pp_consensus"


def _decode_owner():
    return decode_owner_shell(
        prealloc_queue=SimpleNamespace(
            prefetch_prefill_dp_rank_queries=Mock(), queue=[]
        ),
        # Intake has no queued transfers here.
        intake=Mock(return_value=None),
    )


class _BeforeModelForward(Exception):
    pass


# The scheduler truncates once per recorded step, not once over the total.
def total_us(intervals):
    return sum(int(interval * 1e6) for interval in intervals)


class TestSchedulerIdleStepCounters(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="scheduler"))

    @parameterized.expand(
        [
            (
                "prefill_normal",
                Scheduler.event_loop_normal_disagg_prefill,
                ForwardMode.EXTEND,
            ),
            (
                "prefill_overlap",
                Scheduler.event_loop_overlap_disagg_prefill,
                ForwardMode.EXTEND,
            ),
            (
                "prefill_polling",
                Scheduler.event_loop_overlap_disagg_prefill,
                ForwardMode.EXTEND,
            ),
            (
                "decode_normal",
                Scheduler.event_loop_normal_disagg_decode,
                ForwardMode.DECODE,
            ),
            (
                "decode_overlap",
                Scheduler.event_loop_overlap_disagg_decode,
                ForwardMode.DECODE,
            ),
        ]
    )
    def test_disagg_idle_gap(self, name, event_loop, mode):
        # One no-batch iteration flushes an overlap result without on_idle;
        # two iterations also exercise the fully idle branch.
        for idle_iterations in (0, 1, 2):
            with self.subTest(idle_iterations=idle_iterations):
                batches = self.make_batches(mode)
                schedule = batches[:2] + [None] * idle_iterations + batches[2:] + [None]
                poll_pending_copy = name == "prefill_polling"
                if poll_pending_copy and idle_iterations:
                    # Keep work unavailable through the extra unfinished-copy pass.
                    schedule.insert(2, None)
                scheduler = self.make_scheduler(schedule)
                if mode == ForwardMode.DECODE:
                    scheduler.mode.disaggregation_mode = DisaggregationMode.DECODE
                    scheduler.parts.mode_owners.disagg_prefill = None
                    scheduler.parts.mode_owners.disagg_decode = _decode_owner()
                    scheduler.parts.decode_offload_manager = None
                if poll_pending_copy:
                    # Each pending copy adds one intake pass before completing.
                    scheduler.parts.request_receiver.recv_requests.side_effect = [
                        []
                    ] * (len(schedule) + len(batches)) + [StopIteration]
                with (
                    envs.SGLANG_ENABLE_DISAGG_PREFILL_CONTINUOUS_INPUT_POLLING.override(
                        poll_pending_copy
                    )
                ):
                    self.run_and_check(
                        scheduler,
                        event_loop,
                        mode,
                        idle_iterations > 0,
                        poll_pending_copy,
                    )

    def test_polling_admission_miss_during_copy_wait_preserves_busy_interval(self):
        batches = self.make_batches(ForwardMode.EXTEND)
        # New input after a miss can launch while the second copy is unfinished.
        schedule = batches[:2] + [None] + batches[2:] + [None]
        scheduler = self.make_scheduler(schedule)
        inputs = [[]] * (len(schedule) + len(batches))
        inputs[4] = [object()]
        scheduler.parts.request_receiver.recv_requests.side_effect = inputs + [
            StopIteration
        ]
        with envs.SGLANG_ENABLE_DISAGG_PREFILL_CONTINUOUS_INPUT_POLLING.override(True):
            self.run_and_check(
                scheduler,
                Scheduler.event_loop_overlap_disagg_prefill,
                ForwardMode.EXTEND,
                after_idle=False,
                poll_pending_copy=True,
            )

    @parameterized.expand(
        [
            ("unified", Scheduler.event_loop_pp, ForwardMode.EXTEND),
            ("prefill", Scheduler.event_loop_pp, ForwardMode.EXTEND),
            ("decode", Scheduler.event_loop_pp, ForwardMode.DECODE),
        ]
    )
    @unittest.skip("pipeline parallelism is fenced; its ring driver comes later")
    def test_pp_idle_cycles(self, name, event_loop, mode):
        for depth in (0, 1):
            loop_size = 2 + depth
            for pending_transfer in (False,) if name == "unified" else (False, True):
                for pattern in ("consecutive", "empty_slot", "idle_cycle"):
                    with self.subTest(
                        depth=depth, pending_transfer=pending_transfer, pattern=pattern
                    ):
                        batches = self.make_batches(mode)
                        padding = [None] * (loop_size - 2)
                        schedule = batches[:2] + padding
                        if pattern == "idle_cycle":
                            schedule += [None] * loop_size
                        schedule += batches[2:] + padding
                        if pattern == "empty_slot":
                            schedule = [
                                slot
                                for batch in batches
                                for slot in [batch] + [None] * (loop_size - 1)
                            ]
                        schedule += [None] * loop_size  # Drain the last result.
                        scheduler = self.make_scheduler(schedule)
                        self.prepare_pp_scheduler(scheduler)
                        if pending_transfer:
                            if name == "prefill":
                                scheduler.disagg_prefill.transfer = [object()]
                            else:
                                scheduler.disagg_decode.transfer_queue.queue = [
                                    object()
                                ]
                        with (
                            get_parallel().override(
                                pp_size=2,
                                pp_async_batch_depth=depth,
                            ),
                            patch(
                                f"{PD_RING_MODULE}.get_disagg",
                                return_value=SimpleNamespace(
                                    disaggregation_decode_enable_offload_kvcache=False
                                ),
                            ),
                            patch(f"{PP_MODULE}.set_time_batch"),
                        ):
                            self.run_and_check(
                                scheduler, event_loop, mode, pattern == "idle_cycle"
                            )
                        # Transfers suppress housekeeping, not the idle flag.
                        self.assertEqual(
                            scheduler.on_idle.call_count,
                            0 if pending_transfer else 1 + (pattern == "idle_cycle"),
                        )

    @unittest.skip("PD multiplexing is fenced; its loop and split prefill come later")
    def test_split_prefill_is_charged_once_from_its_first_chunk(self):
        # A split prefill spans several run_batch calls but yields one result,
        # so only its first chunk may set the burst's start and its idle flag.
        scheduler = self.make_scheduler([])
        scheduler._sched_idled = True
        batch = self.make_batch(ForwardMode.SPLIT_PREFILL)
        for split_index, launch_ts in enumerate((0.0, 0.2, 0.4)):
            batch.split_index = split_index
            self.launch_batch(scheduler, batch, launch_ts)

        self.assertEqual(scheduler.parts.counters.forward_ct, 3)
        self.assertEqual(batch.forward_iter, 3)
        self.assertEqual(batch.split_prefill_start, (1, 0.0))
        self.assertTrue(batch.after_idle_gap)

        self.record_result(scheduler, batch, 0.5)
        # One charge for the whole burst: first chunk's launch to the result.
        self.assertEqual(
            scheduler.parts.step_counters.total_prefill_busy_us, total_us([0.5 - 0.0])
        )
        self.assertEqual(
            scheduler.parts.step_counters.total_prefill_uncached_tokens, 1024
        )

    def test_a_prefill_is_not_charged_for_an_overlapping_earlier_one(self):
        # An overlapped launch can precede the previous prefill's result; the
        # span already charged to that prefill must not be charged twice.
        scheduler = self.make_scheduler([])
        prefill = self.make_record(ForwardMode.EXTEND, 1, 0.0)
        decode = self.make_record(ForwardMode.DECODE, 2, 0.5, extend_num_tokens=None)
        next_prefill = self.make_record(ForwardMode.EXTEND, 3, 0.6)
        # Both later batches launch while the first prefill is still in flight.
        self.record_result(scheduler, prefill, 1.0)
        self.record_result(scheduler, decode, 1.1)
        self.record_result(scheduler, next_prefill, 1.5)

        # 0.6 -> 1.0 already belongs to the first prefill, so the second one is
        # charged from that result rather than from its own launch.
        self.assertEqual(
            scheduler.parts.step_counters.total_prefill_busy_us, total_us([1.0, 0.5])
        )
        self.assertEqual(
            scheduler.parts.step_counters.total_prefill_uncached_tokens, 2 * 1024
        )
        # The decode broke contiguity, so it contributes no timing sample.
        self.assertEqual(scheduler.parts.step_counters.decode_moment_totals[0], 0)

    def make_batch(self, mode, forward_iter, *, extend_num_tokens=1024):
        return ScheduleBatch(
            reqs=[Mock(rid="request", finished=Mock(return_value=False))],
            forward_mode=mode,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            forward_iter=forward_iter,
            extend_num_tokens=extend_num_tokens,
        )

    def make_record(self, mode, forward_iter, launch_ts, **kwargs):
        return PendingStep(
            step=self.make_batch(mode, forward_iter, **kwargs),
            result=GenerationBatchResult(),
            launch_ts=launch_ts,
            after_idle_gap=False,
        )

    def make_batches(self, mode):
        # Each batch is built as the next launch, so it carries its iteration.
        return [
            self.make_batch(mode, forward_iter)
            for forward_iter in range(1, len(LAUNCH_TIMESTAMPS) + 1)
        ]

    def run_and_check(
        self, scheduler, event_loop, mode, after_idle, poll_pending_copy=False
    ):
        observed_idle_flags = []
        observed_iters = []
        copy_events = []
        # The launch and the commit read the scheduler clock.
        clock = [0.0]
        scheduled = scheduler.schedule

        def schedule(*args, **kwargs):
            batch = scheduled()
            if batch is not None:
                clock[0] = LAUNCH_TIMESTAMPS[batch.forward_iter - 1]
            return batch

        def run_batch(batch, pp_proxy_tensors=None):
            # Exercise the real iteration handoff up to the model forward.
            with self.assertRaises(_BeforeModelForward):
                StepLauncher.run_batch(
                    scheduler.mode.step_launcher, batch, pp_proxy_tensors
                )
            copy_done = Mock(
                query=Mock(
                    return_value=True,
                    side_effect=[False, True] if poll_pending_copy else None,
                )
            )
            copy_events.append(copy_done)
            return GenerationBatchResult(copy_done=copy_done)

        def process_batch_result(batch, result):
            clock[0] = RESULT_TIMESTAMPS[batch.forward_iter - 1]

        step_counters = scheduler.parts.step_counters

        def record_step_counters(record):
            observed_idle_flags.append(record.after_idle_gap)
            observed_iters.append(record.step.forward_iter)
            step_counters.record(record)

        scheduler.schedule_step = schedule
        scheduler.mode.step_launcher.run_batch = run_batch
        scheduler.process_batch_result = process_batch_result
        scheduler.parts.step_counters = SimpleNamespace(record=record_step_counters)
        scheduler.parts.batch_result_processor = Mock()
        scheduler.mode.stages = init_mode_stages(
            scheduler.parts,
            scheduler.loop_schedules(),
            scheduler.hub.request_registry.lora_ids_in_use,
        )
        with (
            get_parallel().override(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
            patch(
                "sglang.srt.managers.scheduler.time.monotonic",
                side_effect=lambda: clock[0],
            ),
            self.assertRaises(StopIteration),
        ):
            event_loop(scheduler)

        if poll_pending_copy:
            self.assertTrue(all(event.query.call_count > 1 for event in copy_events))
        self.assertEqual(observed_idle_flags, [False, False, after_idle, False])
        self.assertEqual(scheduler.parts.counters.forward_ct, 4)
        self.assertEqual(observed_iters, [1, 2, 3, 4])
        if mode == ForwardMode.EXTEND:
            # A prefill is charged from the previous prefill's result, or from
            # its own launch when none applies -- the first one, or after a gap.
            expected_intervals = [
                RESULT_TIMESTAMPS[0] - LAUNCH_TIMESTAMPS[0],
                RESULT_TIMESTAMPS[1] - RESULT_TIMESTAMPS[0],
                RESULT_TIMESTAMPS[2]
                - (LAUNCH_TIMESTAMPS[2] if after_idle else RESULT_TIMESTAMPS[1]),
                RESULT_TIMESTAMPS[3] - RESULT_TIMESTAMPS[2],
            ]
            self.assertEqual(
                step_counters.total_prefill_busy_us, total_us(expected_intervals)
            )
            # Every prefill contributes its tokens; only the span is gap-aware.
            self.assertEqual(
                step_counters.total_prefill_uncached_tokens,
                len(LAUNCH_TIMESTAMPS) * 1024,
            )
        else:
            # Decode keeps launch-to-launch cadence and drops non-contiguous steps.
            expected_intervals = [
                LAUNCH_TIMESTAMPS[1] - LAUNCH_TIMESTAMPS[0],
                LAUNCH_TIMESTAMPS[3] - LAUNCH_TIMESTAMPS[2],
            ]
            if not after_idle:
                expected_intervals.append(LAUNCH_TIMESTAMPS[2] - LAUNCH_TIMESTAMPS[1])
            self.assertEqual(
                step_counters.decode_moment_totals[0], len(expected_intervals)
            )
            self.assertEqual(
                step_counters.decode_moment_totals[2], total_us(expected_intervals)
            )

    def launch_batch(self, scheduler, batch, launch_ts, *, pp_proxy_tensors=None):
        # Only model execution is stopped, at the scripted pre-forward hook.
        with patch(
            "sglang.srt.managers.scheduler.time.monotonic", return_value=launch_ts
        ):
            with self.assertRaises(_BeforeModelForward):
                StepLauncher.run_batch(
                    scheduler.mode.step_launcher, batch, pp_proxy_tensors
                )

    def record_result(self, scheduler, record, result_ts):
        # Prefill accounting reads the clock again when the result lands.
        with patch(
            "sglang.srt.managers.scheduler.time.monotonic", return_value=result_ts
        ):
            scheduler.parts.step_counters.record(record)

    def make_scheduler(self, schedule):
        scheduler = partial_scheduler()
        scheduler.parts.mode_owners = ModeOwners()
        scheduler.parts.lists = CoreLists()
        scheduler.parts.counters = LaunchCounters()
        scheduler.parts.flags = EngineFlags()
        scheduler.tp_size = get_parallel().tp_size
        scheduler.parts.flags.gracefully_exit = False
        scheduler.flat_slot = ScheduleSlot()
        scheduler.parts.flags._engine_paused = False
        scheduler._deferred_input_requests = []
        scheduler._sched_idled = False
        scheduler.parts.step_counters = StepCounters()
        scheduler.parts.counters.forward_ct = 0
        scheduler.parts.counters.processed_tokens_counter = 0
        scheduler.parts.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.enable_overlap = True
        scheduler.parts.is_generation = True
        scheduler.parts.mode_owners.dllm = None
        scheduler.parts.mode_owners.hisparse_coordinator = None
        scheduler.mode.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.hub.request_registry = SimpleNamespace(
            poll_timeout_aborts=Mock(return_value=[]), lora_ids_in_use=set
        )
        scheduler.scheduler_stage_metrics = None
        scheduler.parts.metrics_sink = SimpleNamespace(record_scheduler_active=Mock())
        # The launch runs up to the scripted pre-forward hook.
        launcher = scheduler.mode.step_launcher = StepLauncher.__new__(StepLauncher)
        launcher.counters = scheduler.parts.counters
        launcher.scheduler_stage_metrics = None
        launcher.metrics_sink = scheduler.parts.metrics_sink
        launcher.scripted_scheduler_hook = SimpleNamespace(
            on_run_batch=Mock(side_effect=_BeforeModelForward)
        )
        launcher.launch_batch_sample_if_needed = Mock()
        scheduler.parts.lists.decoding = []
        scheduler.parts.pending = PendingSteps()
        scheduler.parts.request_retirer = make_request_retirer(
            pending=scheduler.parts.pending
        )
        scheduler.parts.recv_skipper = None
        scheduler.parts.lists.chunked_req = None
        scheduler.parts.lists.waiting_queue = []
        scheduler.parts.request_receiver = SimpleNamespace(
            recv_requests=Mock(side_effect=[[]] * len(schedule) + [StopIteration])
        )
        scheduler.process_input_requests = Mock()
        scheduler.parts.mode_owners.disagg_prefill = prefill_owner_shell(
            bootstrap_queue=SimpleNamespace(pop_bootstrapped=Mock(return_value=[])),
            enable_staging=False,
            tp_size=scheduler.tp_size,
            # The KV transfer poll and the chunk pass have no work here.
            after_commit=lambda waiting, rids_to_check=None: waiting,
            process_prefill_chunk=lambda chunked_req, waiting, pending: (
                chunked_req,
                waiting,
            ),
        )
        scheduler.parts.mode_owners.disagg_decode = None
        scheduled_batches = iter(schedule)
        # Intake ends the test; extra admission attempts simply find no work.
        scheduler.schedule = lambda: next(scheduled_batches, None)
        scheduler.is_disable_overlap_for_batch = Mock(return_value=False)
        scheduler._apply_war_barrier = Mock()
        scheduler.on_idle = Mock()
        return scheduler

    def prepare_pp_scheduler(self, scheduler):
        enter_scope(self, get_parallel().override(pp_size=2, pp_rank=0))
        scheduler.pp_group = SimpleNamespace(is_last_rank=True)
        scheduler.parts.streams.forward_stream = Mock()
        scheduler.parts.streams.schedule_stream = Mock()
        scheduler.parts.streams = Streams(
            device="cpu",
            device_module=SimpleNamespace(
                Event=Mock, current_stream=Mock(), stream=lambda stream: nullcontext()
            ),
            schedule_stream=scheduler.parts.streams.schedule_stream,
            forward_stream=scheduler.parts.streams.forward_stream,
            copy_stream=None,
            async_d2h_worker=None,
        )
        scheduler.parts.kv = SimpleNamespace(token_to_kv_pool_allocator=None)
        scheduler.world_group = scheduler.attn_tp_group = None
        scheduler.parts.streams.copy_stream = None
        scheduler.parts.kv.req_to_token_pool = (
            scheduler.parts.kv.token_to_kv_pool_allocator
        ) = None
        scheduler.parts.workers.tp_worker = scheduler.parts.workers.model_worker = (
            scheduler.parts.future_map
        ) = None
        scheduler.parts.pp_spec_forward = None
        enter_scope(self, patch("sglang.srt.managers.scheduler_assembly.PPChannel"))
        enter_scope(
            self,
            patch(f"{PP_MODULE}.PPRing._prepare_tensor_dict", return_value={}),
        )
        enter_scope(
            self,
            patch(
                f"{PP_MODULE}.PPRing."
                "_commit_send_output_work_and_preprocess_output_tensors",
                return_value=(None, GenerationBatchResult(), Mock()),
            ),
        )
        # The PD consensus carries no ids here.
        for target, value in (
            ("_DisaggPrefillRing._get_transferred_ids", None),
            ("_DisaggDecodeRing._get_retract_ids", None),
            ("_DisaggDecodeRing._get_prealloc_ids", None),
            ("_DisaggDecodeRing._get_transferred_ids", None),
            ("_send_consensus_bootstrapped_ids", ([], None)),
            ("_send_consensus_release_ids", ([], None)),
        ):
            enter_scope(self, patch(f"{PD_RING_MODULE}.{target}", return_value=value))
        scheduler._process_hicache_events = Mock()
        scheduler.parts.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: batch
        )
        scheduler.parts.mode_owners.disagg_decode = _decode_owner()
        scheduler.parts.decode_offload_manager = None


if __name__ == "__main__":
    unittest.main()
