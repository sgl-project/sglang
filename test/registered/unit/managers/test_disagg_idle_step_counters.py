"""Prefill busy time and decode timing samples across idle gaps and pauses."""

import importlib.util
import sys
import unittest
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from parameterized import parameterized

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import (
    metrics_reporter as metrics_reporter_module,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.managers.scheduler_components.pause_state import PauseState
from sglang.srt.managers.scheduler_components.prefill_busy_clock import (
    PrefillBusyClock,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.observability.metrics_collector import SchedulerStats
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    published_topology,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

LAUNCH_TIMESTAMPS = (0.0, 0.125, 1.0, 1.125)
# Prefill busy time is charged launch -> result, so the result clock matters too.
RESULT_TIMESTAMPS = tuple(ts + 0.1 for ts in LAUNCH_TIMESTAMPS)
PP_MODULE = "sglang.srt.managers.scheduler_pp_mixin"
SECOND_NS = 1_000_000_000


class _BeforeModelForward(Exception):
    pass


# The scheduler truncates once per recorded step, not once over the total.
def total_us(intervals):
    return sum(int(interval * 1e6) for interval in intervals)


def load_mlx_scheduler_module():
    # Run the real Python loop on CPU without importing a Metal runtime. Use a
    # private module name so this cannot replace the installed MLX scheduler.
    path = (
        Path(scheduler_module.__file__).parents[1]
        / "hardware_backend/mlx/scheduler_mixin.py"
    )
    spec = importlib.util.spec_from_file_location("_idle_counter_mlx_scheduler", path)
    module = importlib.util.module_from_spec(spec)
    core = ModuleType("mlx.core")
    mlx = ModuleType("mlx")
    mlx.core = core
    with patch.dict(sys.modules, {spec.name: module, "mlx": mlx, "mlx.core": core}):
        spec.loader.exec_module(module)
    return module


class _Request(SimpleNamespace):
    # Requests are tracked by identity.
    __hash__ = object.__hash__


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
                if poll_pending_copy:
                    # Each pending copy adds one intake pass before completing.
                    scheduler.request_receiver.recv_requests.side_effect = [[]] * (
                        len(schedule) + len(batches)
                    ) + [StopIteration]
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
        scheduler.request_receiver.recv_requests.side_effect = inputs + [StopIteration]
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
            ("prefill", Scheduler.event_loop_pp_disagg_prefill, ForwardMode.EXTEND),
            ("decode", Scheduler.event_loop_pp_disagg_decode, ForwardMode.DECODE),
        ]
    )
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
                        if name == "unified":
                            scheduler.disaggregation_mode = DisaggregationMode.NULL
                        self.prepare_pp_scheduler(scheduler)
                        if pending_transfer:
                            if name == "prefill":
                                scheduler.disagg_prefill_transfer_queue = [object()]
                            else:
                                scheduler.disagg_decode_transfer_queue.queue = [
                                    object()
                                ]
                        with (
                            get_parallel().override(
                                pp_size=2,
                                pp_async_batch_depth=depth,
                            ),
                            patch(
                                f"{PP_MODULE}.get_disagg",
                                return_value=SimpleNamespace(
                                    disaggregation_decode_enable_offload_kvcache=False
                                ),
                            ),
                            patch(f"{PP_MODULE}.set_time_batch"),
                        ):
                            self.run_and_check(
                                scheduler, event_loop, mode, pattern == "idle_cycle"
                            )
                        # Each launched micro-batch gets its write-after-read barrier.
                        self.assertEqual(scheduler._apply_war_barrier.call_count, 4)
                        # Transfers suppress housekeeping, not the idle flag.
                        self.assertEqual(
                            scheduler.on_idle.call_count,
                            0 if pending_transfer else 1 + (pattern == "idle_cycle"),
                        )

    @parameterized.expand([("fresh", False), ("chained", True)])
    def test_mlx_idle_gap(self, name, chained):
        module = load_mlx_scheduler_module()
        mixin = module.SchedulerMlxOverlapMixin
        mode = ForwardMode.DECODE if chained else ForwardMode.EXTEND
        for idle_iterations in (0, 1, 2):
            with self.subTest(idle_iterations=idle_iterations):
                batches = self.make_batches(mode)
                for batch in batches:
                    batch.sampling_info = Mock()
                if chained:
                    # Finishing the request drains its already-launched lookahead
                    # before scheduling fresh work, allowing the loop to go idle.
                    for batch in batches:
                        batch.reqs[0].finished.return_value = True
                    schedule = (
                        [batches[0]] + [None] * idle_iterations + [batches[2], None]
                    )
                else:
                    schedule = (
                        batches[:2] + [None] * idle_iterations + batches[2:] + [None]
                    )
                scheduler = self.make_scheduler(schedule)
                scheduler.disaggregation_mode = DisaggregationMode.NULL
                scheduler.scripted_scheduler_hook = None
                scheduler.future_map = None
                scheduler.result_queue = deque()
                scheduler._prepare_mlx_launch = MethodType(
                    mixin._prepare_mlx_launch, scheduler
                )
                scheduler._finalize_mlx_pending_job = MethodType(
                    mixin._finalize_mlx_pending_job, scheduler
                )
                scheduler._mlx_batch_chain_safe = MethodType(
                    mixin._mlx_batch_chain_safe, scheduler
                )
                launch = SimpleNamespace(
                    mode="decode" if chained else "extend",
                    decode=object() if chained else None,
                )
                scheduler.tp_worker = SimpleNamespace(
                    async_forward_batch_generation_mlx=Mock(return_value=launch),
                    async_chained_decode_mlx=Mock(return_value=launch),
                    finalize_mlx_result=Mock(return_value=GenerationBatchResult()),
                )
                with (
                    patch.object(module, "resolve_forward_inputs"),
                    patch.object(
                        module,
                        "get_device",
                        return_value=SimpleNamespace(mlx_enable_sampling=False),
                    ),
                    patch.object(
                        scheduler_module.time,
                        "monotonic",
                        side_effect=LAUNCH_TIMESTAMPS,
                    ),
                ):
                    self.run_and_check(
                        scheduler,
                        mixin.event_loop_overlap_mlx,
                        mode,
                        idle_iterations > 0,
                    )
                self.assertEqual(
                    scheduler.tp_worker.async_chained_decode_mlx.call_count,
                    2 if chained else 0,
                )

    def test_mlx_pending_job_preserves_inflight_request_inventory(self):
        module = load_mlx_scheduler_module()
        scheduler = self.make_scheduler([])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        req = Req("mlx-pending", "", [1], SamplingParams())
        batch = ScheduleBatch(reqs=[req], forward_mode=ForwardMode.DECODE)
        scheduler.last_batch = batch
        pending = module.MlxPendingJob(
            launch=None,
            batch_copy=batch.copy(),
            schedule_batch=batch,
            reqs=batch.reqs[:],
        )
        scheduler.result_queue = deque([pending])
        # The partial scheduler has none of the lists request_lists() reads.
        scheduler.request_lists = lambda: {}

        self.assertEqual(scheduler.collect_inflight_reqs(), {req})

    def test_processed_tokens_count_each_prefill_once(self):
        # The aging counter must not grow when a prefill batch is reused for
        # decode with a stale extend_num_tokens.
        scheduler = self.make_scheduler([])
        batch = self.make_batch(ForwardMode.EXTEND)
        self.launch_batch(scheduler, batch, 0.0)
        # As prepare_for_decode does: switch the mode, keep extend_num_tokens.
        batch.forward_mode = ForwardMode.DECODE
        for launch_ts in (0.1, 0.2, 0.3):
            self.launch_batch(scheduler, batch, launch_ts)
        self.assertEqual(scheduler.processed_tokens_counter, 1024)

    def test_a_prefill_is_not_charged_for_an_overlapping_earlier_one(self):
        # An overlapped launch can precede the previous prefill's result; the
        # span already charged to that prefill must not be charged twice.
        scheduler = self.make_scheduler([])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        prefill = self.make_batch(ForwardMode.EXTEND)
        decode = self.make_batch(ForwardMode.DECODE, extend_num_tokens=None)
        next_prefill = self.make_batch(ForwardMode.EXTEND)
        # Both later batches launch while the first prefill is still in flight.
        self.launch_batch(scheduler, prefill, 0.0)
        self.launch_batch(scheduler, decode, 0.5)
        self.launch_batch(scheduler, next_prefill, 0.6)

        self.record_result(scheduler, prefill, 1.0)
        self.record_result(scheduler, decode, 1.1)
        self.record_result(scheduler, next_prefill, 1.5)

        # 0.6 -> 1.0 already belongs to the first prefill, so the second one is
        # charged from that result rather than from its own launch.
        self.assertEqual(scheduler.total_prefill_busy_us, total_us([1.0, 0.5]))
        self.assertEqual(scheduler.total_prefill_uncached_tokens, 2 * 1024)
        # The decode broke contiguity, so it contributes no timing sample.
        self.assertEqual(scheduler.decode_moment_totals[0], 0)

    def test_long_prefill_step_is_charged_but_long_decode_cadence_is_not(self):
        # A long-context chunk can outlast STEP_MAX_US and is still busy time.
        long_step = scheduler_module.STEP_MAX_US / 1e6 + 1.5
        scheduler = self.make_scheduler([])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        for i in range(2):
            prefill = self.make_batch(ForwardMode.EXTEND)
            self.launch_batch(scheduler, prefill, i * long_step)
            self.record_result(scheduler, prefill, (i + 1) * long_step)

        self.assertEqual(scheduler.total_prefill_busy_us, total_us([long_step] * 2))
        self.assertEqual(scheduler.total_prefill_uncached_tokens, 2 * 1024)

        # A decode or verify launch gap that long is a stall, not a cadence sample.
        for start, mode in ((10, ForwardMode.DECODE), (20, ForwardMode.TARGET_VERIFY)):
            for i in range(2):
                batch = self.make_batch(mode, extend_num_tokens=None)
                self.launch_batch(scheduler, batch, start + i * long_step)
                self.record_result(scheduler, batch, start + i * long_step + 0.1)
        self.assertEqual(scheduler.decode_moment_totals[0], 0)

    def test_pause_between_prefills_is_not_charged(self):
        scheduler = self.make_scheduler([])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        first, second = (self.make_batch(ForwardMode.EXTEND) for _ in range(2))
        self.launch_batch(scheduler, first, 0.0)
        self.record_result(scheduler, first, 0.125)
        self.pause_and_resume(scheduler, 0.125, 10.0)
        self.launch_batch(scheduler, second, 10.0)
        self.record_result(scheduler, second, 10.125)

        self.assertEqual(scheduler.total_prefill_busy_us, total_us([0.125, 0.125]))
        self.assertEqual(scheduler.total_prefill_uncached_tokens, 2 * 1024)

    def test_result_held_across_a_pause_excludes_the_pause(self):
        # An overlapped prefill launched before the pause has its result processed after resume.
        scheduler = self.make_scheduler([])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        batch = self.make_batch(ForwardMode.EXTEND)
        self.launch_batch(scheduler, batch, 0.0)
        self.pause_and_resume(scheduler, 0.05, 10.0)
        self.record_result(scheduler, batch, 10.125)

        self.assertEqual(scheduler.total_prefill_busy_us, int((10.125 - 9.95) * 1e6))

    def test_result_processed_during_an_open_pause_excludes_it(self):
        # A retract after an in-place pause processes the held result before resume.
        scheduler = self.make_scheduler([])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        first, second = (self.make_batch(ForwardMode.EXTEND) for _ in range(2))
        self.launch_batch(scheduler, first, 0.0)
        self.pause(scheduler, 0.125)
        self.record_result(scheduler, first, 10.125)
        self.assertEqual(scheduler.total_prefill_busy_us, total_us([0.125]))

        self.resume(scheduler, 11.0)
        self.launch_batch(scheduler, second, 11.0)
        self.record_result(scheduler, second, 11.125)
        self.assertEqual(scheduler.total_prefill_busy_us, total_us([0.125, 0.125]))

    def test_many_pauses_inside_one_step_are_all_excluded(self):
        scheduler = self.make_scheduler([])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        batch = self.make_batch(ForwardMode.EXTEND)
        self.launch_batch(scheduler, batch, 0.0)
        for i in range(100):
            self.pause_and_resume(scheduler, 1.0 + i * 0.125, 1.0625 + i * 0.125)
        self.record_result(scheduler, batch, 14.0)

        self.assertEqual(scheduler.total_prefill_busy_us, total_us([14.0 - 6.25]))
        # Every window ended before this step's end, so none is kept.
        self.assertEqual(scheduler.pause_state.consume_paused_time(0.0, 14.0), 0.0)

    def test_pause_inside_an_idle_gap_is_not_subtracted_from_the_next_prefill(self):
        scheduler = self.make_scheduler([])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        first, second = (self.make_batch(ForwardMode.EXTEND) for _ in range(2))
        self.launch_batch(scheduler, first, 0.0)
        self.record_result(scheduler, first, 0.125)
        self.pause_and_resume(scheduler, 0.25, 0.5)
        scheduler._sched_idled = True
        self.launch_batch(scheduler, second, 1.0)
        self.record_result(scheduler, second, 1.125)

        self.assertEqual(scheduler.total_prefill_busy_us, total_us([0.125, 0.125]))

    def test_repeated_pause_keeps_one_window_from_the_first_start(self):
        scheduler = self.make_scheduler([])
        self.pause(scheduler, 1.0)
        self.pause(scheduler, 2.0)
        self.resume(scheduler, 3.0)
        # One window, [1.0, 3.0].
        self.assertEqual(scheduler.pause_state.consume_paused_time(0.0, 1.5), 0.5)
        self.assertEqual(scheduler.pause_state.consume_paused_time(0.0, 4.0), 2.0)

    def test_pd_prefill_steps_count_tokens_but_not_step_time(self):
        scheduler = self.make_scheduler([])
        # A long step and a zero-length step both credit their tokens.
        for launch_ts, result_ts in ((0.0, 5.0), (6.0, 6.0)):
            batch = self.make_batch(ForwardMode.EXTEND)
            self.launch_batch(scheduler, batch, launch_ts)
            self.record_result(scheduler, batch, result_ts)
        probe = self.make_batch(ForwardMode.EXTEND)
        probe.reqs[0].rid = f"{HEALTH_CHECK_RID_PREFIX}_probe"
        self.launch_batch(scheduler, probe, 7.0)
        self.record_result(scheduler, probe, 7.1)

        self.assertEqual(scheduler.total_prefill_busy_us, 0)
        self.assertEqual(scheduler.total_prefill_uncached_tokens, 2 * 1024)

    @parameterized.expand(
        [
            ("pd_prefill", DisaggregationMode.PREFILL, 2_000_000),
            ("unified", DisaggregationMode.NULL, 7),
            ("pd_decode", DisaggregationMode.DECODE, 7),
        ]
    )
    def test_load_snapshot_carries_the_busy_clock_only_on_pd_prefill(
        self, name, mode, expected
    ):
        clock = PrefillBusyClock()
        scheduler = SimpleNamespace(
            disaggregation_mode=mode,
            server_args=None,
            max_total_num_tokens=1000,
            kv_shard_widening=1,
            max_running_requests=10,
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(
                    get_kv_token_stats=lambda: (0, 0.0)
                )
            ),
            tp_worker=None,
            token_to_kv_pool_allocator=None,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            tree_cache=SimpleNamespace(
                waiting_storage_span=lambda handle: None,
                storage_metrics_collector=None,
            ),
            policy=SimpleNamespace(waiting_queue_prefix_matched=lambda queue: True),
            metrics_reporter=SimpleNamespace(
                recent_cache_hit_rate=0.0,
                stats=SchedulerStats(),
                prefill_busy_clock=clock,
            ),
            running_batch=SimpleNamespace(reqs=[]),
            waiting_queue=[],
            chunked_req=None,
            request_lists=lambda: {kind: [] for kind in ListKind},
            disagg_decode_prealloc_queue=SimpleNamespace(
                queue={},
                host_pool_blocked=False,
                transfer_queue=SimpleNamespace(num_host_staged_landed=0),
            ),
        )
        scheduler._get_total_prefill_busy_us = MethodType(
            Scheduler._get_total_prefill_busy_us, scheduler
        )
        with published_topology(role="scheduler"):
            Scheduler.init_load_inquirer(scheduler)
            scheduler.total_prefill_busy_us = 7
            clock.mark_busy(SECOND_NS)
            with patch(
                "sglang.srt.managers.scheduler.time.monotonic_ns",
                return_value=3 * SECOND_NS,
            ):
                snapshot = scheduler.load_inquirer.get_loads()

        self.assertEqual(snapshot.total_prefill_busy_us, expected)

    def test_pd_prefill_snapshot_includes_the_step_tokens(self):
        scheduler = self.make_scheduler([])
        published = []
        scheduler.deferred_responses = SimpleNamespace(stream_released=Mock())
        scheduler.publish_load_snapshot = Mock(
            side_effect=lambda force: published.append(
                scheduler.total_prefill_uncached_tokens
            )
        )
        scheduler.load_publisher = SimpleNamespace(publish_load_stat=Mock())
        scheduler.load_inquirer = SimpleNamespace(get_loads=Mock())
        scheduler.process_batch_result_disagg_prefill = Mock()
        scheduler._on_step_processed = Mock()
        scheduler.tree_cache = SimpleNamespace(flush_pending_backups=Mock())
        scheduler.metrics_reporter.log_batch_result_stats = Mock()
        scheduler.metrics_reporter.update_device_timer = Mock()
        scheduler.enable_fpm = False
        scheduler._maybe_clear_mm_inputs = Mock()
        scheduler.maybe_send_health_check_signal = Mock()
        batch = self.make_batch(ForwardMode.EXTEND)
        self.launch_batch(scheduler, batch, 0.0)
        with patch.object(scheduler_module, "flush_trace_batch"):
            Scheduler.process_batch_result(scheduler, batch, GenerationBatchResult())

        self.assertEqual(published, [1024])
        self.assertEqual(scheduler.total_prefill_uncached_tokens, 1024)

    @parameterized.expand([("in_place",), ("retract",), ("pp_retract",)])
    def test_pause_stops_the_pd_prefill_busy_clock_and_resume_restarts_it(self, mode):
        scheduler = self.make_scheduler([])
        scheduler.enable_overlap_mlx = False
        scheduler.decode_offload_manager = None
        scheduler.hisparse_coordinator = None
        scheduler.dllm_config = None
        scheduler.kv_events_publisher = Mock()
        scheduler.metrics_reporter.current_scheduler_metrics_enabled = False
        if mode == "pp_retract":
            enter_scope(self, get_parallel().override(pp_size=2, pp_rank=0))
            # The micro-batch retract has its own tests; this checks the clock.
            scheduler._pp_retract_microbatches = Mock()
        clock = scheduler.metrics_reporter.prefill_busy_clock
        clock.mark_busy(0)
        self.pause(scheduler, 1.0, mode="retract" if mode == "pp_retract" else mode)
        # A paused loop pass with queued work does not reopen the clock.
        clock.mark_busy(2 * SECOND_NS)
        self.resume(scheduler, 5.0)
        self.assertEqual(clock._read_us(6 * SECOND_NS), 2_000_000)

    def test_receive_and_launch_open_the_pd_prefill_busy_clock(self):
        scheduler = self.make_scheduler([])
        scheduler.metrics_reporter = self.make_reporter()
        scheduler.uocr_runtime = None
        scheduler.request_receiver = SimpleNamespace(
            recv_requests=Mock(side_effect=[[], [SimpleNamespace()]])
        )
        clock = scheduler.metrics_reporter.prefill_busy_clock
        # An empty receive at 0.5 s, then one starting at 1 s; later readings are 3 s.
        readings = iter((SECOND_NS // 2, SECOND_NS))
        scheduler_time = SimpleNamespace(
            monotonic_ns=lambda: next(readings, 3 * SECOND_NS)
        )
        with (
            patch.object(metrics_reporter_module, "ENABLE_METRICS_DEVICE_TIMER", False),
            patch.object(scheduler_module, "time", scheduler_time),
        ):
            scheduler.ingest_requests()
            self.assertEqual(clock._read_us(SECOND_NS), 0)
            scheduler.ingest_requests()
        clock._mark_idle(2 * SECOND_NS)
        with (
            patch.object(metrics_reporter_module, "ENABLE_METRICS_DEVICE_TIMER", False),
            patch(
                "sglang.srt.managers.scheduler.time.monotonic_ns",
                return_value=4 * SECOND_NS,
            ),
        ):
            self.launch_batch(scheduler, self.make_batch(ForwardMode.EXTEND), 4.0)
        self.assertEqual(clock._read_us(5 * SECOND_NS), 2_000_000)

    @parameterized.expand(
        [
            ("normal", Scheduler.event_loop_normal_disagg_prefill),
            ("overlap", Scheduler.event_loop_overlap_disagg_prefill),
        ]
    )
    def test_pd_prefill_loop_closes_the_busy_clock_once_fully_idle(
        self, name, event_loop
    ):
        scheduler = self.make_scheduler([None])
        scheduler.__dict__.update(self.make_idle_scheduler().__dict__)
        # Run the loop's real on_idle instead of the fixture's mock.
        del scheduler.on_idle
        scheduler.ingest_requests = Mock()
        scheduler.is_fully_idle = Mock(return_value=True)

        def stop_after_one_pass():
            scheduler.gracefully_exit = True

        scheduler.process_disagg_prefill_transfer_queue = stop_after_one_pass
        clock = scheduler.metrics_reporter.prefill_busy_clock
        clock.mark_busy(0)
        with (
            patch.object(metrics_reporter_module, "ENABLE_METRICS_DEVICE_TIMER", False),
            patch(
                "sglang.srt.managers.scheduler.time.monotonic_ns",
                return_value=3 * SECOND_NS,
            ),
        ):
            event_loop(scheduler)

        self.assertEqual(clock._read_us(10 * SECOND_NS), 3_000_000)

    def test_on_idle_counts_head_and_tail_with_metrics_off(self):
        scheduler = self.make_idle_scheduler()
        clock = scheduler.metrics_reporter.prefill_busy_clock
        clock.mark_busy(0)
        # Tail waiting on decode, fully idle at 3 s, then a 2 s head from 10 s.
        timeline = ((False, 2.5), (True, 3.0), (False, 10.0), (True, 12.0))
        with patch.object(
            metrics_reporter_module, "ENABLE_METRICS_DEVICE_TIMER", False
        ):
            for fully_idle, now_s in timeline:
                scheduler.is_fully_idle = Mock(return_value=fully_idle)
                with patch(
                    "sglang.srt.managers.scheduler.time.monotonic_ns",
                    return_value=int(now_s * SECOND_NS),
                ):
                    scheduler.on_idle()

        self.assertEqual(clock._read_us(20 * SECOND_NS), 5_000_000)

    def test_on_idle_closes_the_clock_before_the_idle_sleep(self):
        scheduler = self.make_idle_scheduler()
        scheduler.is_fully_idle = Mock(return_value=True)
        clock = scheduler.metrics_reporter.prefill_busy_clock
        clock.mark_busy(0)
        now = {"ns": 3 * SECOND_NS}

        def sleep_30_s():
            now["ns"] += 30 * SECOND_NS

        scheduler.maybe_sleep_on_idle = Mock(side_effect=sleep_30_s)
        with (
            patch.object(metrics_reporter_module, "ENABLE_METRICS_DEVICE_TIMER", False),
            patch(
                "sglang.srt.managers.scheduler.time.monotonic_ns",
                side_effect=lambda: now["ns"],
            ),
        ):
            scheduler.on_idle()

        self.assertEqual(clock._read_us(now["ns"]), 3_000_000)

    def make_reporter(self):
        reporter = SimpleNamespace(
            scheduler=SimpleNamespace(enable_fpm=False),
            enable_metrics=False,
            prefill_busy_clock=PrefillBusyClock(),
            _maybe_log_idle_metrics=Mock(),
        )
        for name in (
            "record_scheduler_active",
            "record_scheduler_idle",
            "_record_scheduler_time",
        ):
            method = getattr(SchedulerMetricsReporter, name)
            setattr(reporter, name, MethodType(method, reporter))
        return reporter

    def make_idle_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.metrics_reporter = self.make_reporter()
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.scheduler_stage_metrics = None
        scheduler.dp_attn_adapter = Mock()
        scheduler.deferred_responses = SimpleNamespace(
            stream_released=Mock(), has_pending=Mock(return_value=False)
        )
        scheduler.maybe_send_health_check_signal = Mock()
        scheduler._last_stall_publish_ts = float("-inf")
        scheduler.publish_load_snapshot = Mock(return_value=None)
        scheduler.load_publisher = Mock()
        scheduler.load_inquirer = Mock()
        scheduler.enable_hicache_storage = False
        scheduler.enable_lmcache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.request_receiver = SimpleNamespace(wait_for_progress=Mock())
        scheduler.enable_unified_memory = False
        scheduler.invariant_checker = Mock()
        scheduler.gc_freezer = Mock()
        scheduler.forward_ct = 0
        scheduler.kv_events_publisher = Mock()
        scheduler.new_token_ratio_tracker = Mock()
        scheduler.maybe_sleep_on_idle = Mock()
        return scheduler

    def pause(self, scheduler, paused_at, mode="in_place"):
        with (
            patch(
                "sglang.srt.managers.scheduler_components.pause_state.time.monotonic",
                return_value=paused_at,
            ),
            patch(
                "sglang.srt.managers.scheduler.time.monotonic_ns",
                return_value=int(paused_at * SECOND_NS),
            ),
        ):
            Scheduler.pause_generation(scheduler, PauseGenerationReqInput(mode=mode))

    def resume(self, scheduler, resumed_at):
        with (
            patch(
                "sglang.srt.managers.scheduler_components.pause_state.time.monotonic",
                return_value=resumed_at,
            ),
            patch(
                "sglang.srt.managers.scheduler.time.monotonic_ns",
                return_value=int(resumed_at * SECOND_NS),
            ),
        ):
            Scheduler.continue_generation(
                scheduler, ContinueGenerationReqInput(torch_empty_cache=False)
            )

    def pause_and_resume(self, scheduler, paused_at, resumed_at):
        self.pause(scheduler, paused_at)
        self.resume(scheduler, resumed_at)

    def make_batch(self, mode, *, launch_ts=None, extend_num_tokens=1024):
        return ScheduleBatch(
            reqs=[
                _Request(
                    rid="request", finished=Mock(return_value=False), inflight_steps=0
                )
            ],
            forward_mode=mode,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            launch_ts=launch_ts,
            extend_num_tokens=extend_num_tokens,
        )

    def make_batches(self, mode):
        return [
            self.make_batch(mode, launch_ts=launch_ts)
            for launch_ts in LAUNCH_TIMESTAMPS
        ]

    def run_and_check(
        self, scheduler, event_loop, mode, after_idle, poll_pending_copy=False
    ):
        observed_idle_flags = []
        observed_iters = []
        copy_events = []

        def run_batch(batch, pp_proxy_tensors=None):
            # Exercise the real timestamp, iteration, and flag handoff.
            self.launch_batch(
                scheduler, batch, batch.launch_ts, pp_proxy_tensors=pp_proxy_tensors
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
            observed_idle_flags.append(batch.after_idle_gap)
            observed_iters.append(batch.forward_iter)
            self.record_result(
                scheduler,
                batch,
                RESULT_TIMESTAMPS[batch.forward_iter - 1],
                result=result,
            )

        scheduler.run_batch = run_batch
        scheduler.process_batch_result = process_batch_result
        with (
            get_parallel().override(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
            self.assertRaises(StopIteration),
        ):
            event_loop(scheduler)

        if poll_pending_copy:
            self.assertTrue(all(event.query.call_count > 1 for event in copy_events))
        self.assertEqual(observed_idle_flags, [False, False, after_idle, False])
        self.assertEqual(scheduler.forward_ct, 4)
        self.assertEqual(observed_iters, [1, 2, 3, 4])
        if (
            mode == ForwardMode.EXTEND
            and scheduler.disaggregation_mode == DisaggregationMode.PREFILL
        ):
            # PD prefill busy time is the busy clock; steps only credit tokens.
            self.assertEqual(scheduler.total_prefill_busy_us, 0)
            self.assertEqual(
                scheduler.total_prefill_uncached_tokens,
                len(LAUNCH_TIMESTAMPS) * 1024,
            )
        elif mode == ForwardMode.EXTEND:
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
                scheduler.total_prefill_busy_us, total_us(expected_intervals)
            )
            # Every prefill contributes its tokens; only the span is gap-aware.
            self.assertEqual(
                scheduler.total_prefill_uncached_tokens,
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
            self.assertEqual(scheduler.decode_moment_totals[0], len(expected_intervals))
            self.assertEqual(
                scheduler.decode_moment_totals[2], total_us(expected_intervals)
            )

    def launch_batch(self, scheduler, batch, launch_ts, *, pp_proxy_tensors=None):
        # Only model execution is stopped, at the scripted pre-forward hook.
        with patch(
            "sglang.srt.managers.scheduler.time.monotonic", return_value=launch_ts
        ):
            with self.assertRaises(_BeforeModelForward):
                Scheduler.run_batch(scheduler, batch, pp_proxy_tensors)

    def record_result(self, scheduler, batch, result_ts, *, result=None):
        # Prefill accounting reads the clock again when the result lands.
        with patch(
            "sglang.srt.managers.scheduler.time.monotonic", return_value=result_ts
        ):
            scheduler._record_step_counters(
                batch, GenerationBatchResult() if result is None else result
            )

    def make_scheduler(self, schedule):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_reqs = KeyedReqList(ListKind.DECODING)
        scheduler.prefill_launched_reqs = KeyedReqList(ListKind.PREFILL_LAUNCHED)
        scheduler.tp_size = get_parallel().tp_size
        scheduler.dp_tp_group = SimpleNamespace(
            world_size=get_parallel().attn_tp_size
            if get_parallel().attn_dp_enabled
            else get_parallel().tp_size
        )
        scheduler.gracefully_exit = False
        scheduler.pause_state = PauseState()
        scheduler._sched_idled = False
        scheduler._prev_step = None
        scheduler._prev_prefill_end_ts = None
        scheduler.forward_ct = 0
        scheduler.processed_tokens_counter = 0
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.enable_overlap = True
        scheduler.enable_continuous_input_polling = False
        scheduler.is_generation = True
        scheduler.dllm_config = None
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler._poll_timeout_aborts = Mock(return_value=[])
        scheduler.scheduler_stage_metrics = None
        scheduler.metrics_reporter = SimpleNamespace(
            record_scheduler_active=Mock(), prefill_busy_clock=PrefillBusyClock()
        )
        scheduler.invariant_checker = Mock()
        scheduler.scripted_scheduler_hook = SimpleNamespace(
            on_run_batch=Mock(side_effect=_BeforeModelForward)
        )
        scheduler.profiler_manager = SimpleNamespace(_profile_batch_predicate=Mock())
        scheduler.total_prefill_busy_us = 0
        scheduler.total_prefill_uncached_tokens = 0
        scheduler.decode_moment_totals = [0.0] * 6
        scheduler.running_batch = ScheduleBatch(reqs=[])
        scheduler.last_batch = None
        scheduler.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        scheduler._pending_chunked_abort_req = None
        scheduler.waiting_queue = IndexedReqList(ListKind.WAITING)
        scheduler.disagg_prefill_transfer_queue = []
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[], staging_handler=None
        )
        scheduler.enable_staging = False
        scheduler.request_receiver = SimpleNamespace(
            recv_requests=Mock(side_effect=[[]] * len(schedule) + [StopIteration])
        )
        scheduler.process_input_requests = Mock()
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            move_bootstrapped=Mock(return_value=[])
        )
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            prefetch_prefill_dp_rank_queries=Mock(), queue=[]
        )
        scheduler.process_decode_queue = Mock(return_value=False)
        scheduler.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, chunked_req: batch
        )
        scheduled_batches = iter(schedule)
        # Intake ends the test; extra admission attempts simply find no work.
        next_plan = Mock(
            side_effect=lambda *args, **kwargs: SimpleNamespace(
                running_batch=scheduler.running_batch,
                batch_to_run=next(scheduled_batches, None),
            )
        )
        scheduler.get_next_disagg_prefill_batch_to_run = next_plan
        scheduler.get_next_disagg_decode_batch_to_run = next_plan
        scheduler.get_next_batch_to_run = next_plan
        scheduler.get_new_batch_prefill = next_plan
        scheduler.should_process_last_batch_early = Mock(return_value=False)
        scheduler._war_barrier_is_coarse = Mock(return_value=True)
        scheduler._apply_war_barrier = Mock()
        scheduler.process_disagg_prefill_transfer_queue = Mock()
        scheduler.launch_batch_sample_if_needed = Mock()
        scheduler.on_idle = Mock()
        return scheduler

    def prepare_pp_scheduler(self, scheduler):
        enter_scope(self, get_parallel().override(pp_size=2, pp_rank=0))
        scheduler.pp_group = SimpleNamespace(is_last_rank=True)
        scheduler.token_to_kv_pool_allocator = None
        scheduler.forward_stream_ctx = nullcontext()
        scheduler.forward_stream = Mock()
        scheduler.schedule_stream = Mock()
        scheduler.device_module = SimpleNamespace(Event=Mock, current_stream=Mock())
        scheduler._pp_recv_proxy_tensors = Mock(return_value=None)
        scheduler._pp_commit_comm_work = Mock()
        scheduler._pp_prepare_tensor_dict = Mock(return_value={})
        scheduler._pp_commit_send_output_work_and_preprocess_output_tensors = Mock(
            return_value=(None, GenerationBatchResult(), Mock())
        )
        scheduler._pp_pd_get_retract_ids = Mock(return_value=None)
        scheduler._pp_pd_get_prealloc_ids = Mock(return_value=None)
        scheduler._pp_pd_get_decode_transferred_ids = Mock(return_value=None)
        scheduler._pp_pd_send_consensus_bootstrapped_ids = Mock(return_value=([], None))
        scheduler._pp_pd_send_consensus_release_ids = Mock(return_value=([], None))
        scheduler.process_prefill_chunk = Mock()
        scheduler._process_hicache_events = Mock()
        scheduler.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: batch
        )
        scheduler.disagg_prefill_transfer_queue = []
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[], staging_handler=None
        )


class TestPrefillBusyClock(CustomTestCase):
    def test_head_tail_and_long_spans_count_but_fully_idle_does_not(self):
        clock = PrefillBusyClock()
        # A burst received at 0 whose tail waits on decode until 3 s.
        clock.mark_busy(0)
        clock.mark_busy(int(0.05 * SECOND_NS))
        self.assertEqual(clock._read_us(int(1.5 * SECOND_NS)), 1_500_000)
        clock._mark_idle(3 * SECOND_NS)
        self.assertEqual(clock._read_us(10 * SECOND_NS), 3_000_000)
        # A head received at 10 s that waits on L3 before its forward.
        clock.mark_busy(10 * SECOND_NS)
        clock._mark_idle(13 * SECOND_NS)
        self.assertEqual(clock._read_us(20 * SECOND_NS), 6_000_000)

    def test_a_backdated_receive_does_not_reopen_closed_time(self):
        clock = PrefillBusyClock()
        clock.mark_busy(0)
        clock._mark_idle(5 * SECOND_NS)
        clock.mark_busy(4 * SECOND_NS)
        clock._mark_idle(6 * SECOND_NS)
        self.assertEqual(clock._read_us(6 * SECOND_NS), 6_000_000)

    def test_pause_closes_the_span_and_resume_reopens_it(self):
        clock = PrefillBusyClock()
        clock.mark_busy(0)
        clock._pause(2 * SECOND_NS)
        clock.mark_busy(3 * SECOND_NS)
        self.assertEqual(clock._read_us(4 * SECOND_NS), 2_000_000)
        clock._resume(5 * SECOND_NS)
        self.assertEqual(clock._read_us(6 * SECOND_NS), 3_000_000)

    def test_restart_drops_counted_time_and_keeps_a_pause(self):
        clock = PrefillBusyClock()
        clock.mark_busy(0)
        clock._pause(SECOND_NS)
        clock._restart(7, 2 * SECOND_NS)
        clock.mark_busy(3 * SECOND_NS)
        self.assertEqual(clock._read_us(4 * SECOND_NS), 7)
        clock._resume(5 * SECOND_NS)
        self.assertEqual(clock._read_us(6 * SECOND_NS), 1_000_007)

    def test_a_receive_from_before_a_restart_counts_from_the_restart(self):
        clock = PrefillBusyClock()
        clock._restart(7, 2 * SECOND_NS)
        clock.mark_busy(SECOND_NS)
        self.assertEqual(clock._read_us(3 * SECOND_NS), 1_000_007)

    def test_reads_never_decrease(self):
        clock = PrefillBusyClock()
        clock.mark_busy(0)
        reads = [clock._read_us(SECOND_NS)]
        clock._mark_idle(2 * SECOND_NS)
        reads.append(clock._read_us(2 * SECOND_NS))
        clock._pause(3 * SECOND_NS)
        reads.append(clock._read_us(3 * SECOND_NS))
        clock._resume(3 * SECOND_NS)
        clock._mark_idle(3 * SECOND_NS)
        # A receive stamped before the last close counts only from that close.
        clock.mark_busy(SECOND_NS)
        reads.append(clock._read_us(4 * SECOND_NS))
        self.assertEqual(reads, [1_000_000, 2_000_000, 2_000_000, 3_000_000])

    def test_public_methods_read_the_current_time(self):
        clock = PrefillBusyClock()
        with patch(
            "sglang.srt.managers.scheduler_components.prefill_busy_clock.time.monotonic_ns"
        ) as now:
            now.return_value = SECOND_NS
            clock.restart(7)
            # A receive stamped before the restart counts from the restart.
            clock.mark_busy(0)
            now.return_value = 2 * SECOND_NS
            clock.pause()
            now.return_value = 4 * SECOND_NS
            clock.resume()
            now.return_value = 5 * SECOND_NS
            reads = now.call_count
            # mark_idle closes and returns the same single read.
            self.assertEqual(clock.mark_idle(), 5 * SECOND_NS)
            self.assertEqual(now.call_count, reads + 1)
            clock.mark_busy(6 * SECOND_NS)
            now.return_value = 7 * SECOND_NS
            self.assertEqual(clock.read_us(), 3_000_007)

    def test_a_span_after_the_clock_moves_back_counts_in_full(self):
        clock = PrefillBusyClock()
        clock.mark_busy(0)
        clock._mark_idle(100 * SECOND_NS)
        # A restore onto a host with less uptime moves the clock back to 10 s.
        clock._mark_idle(10 * SECOND_NS)
        clock.mark_busy(11 * SECOND_NS)
        clock._mark_idle(14 * SECOND_NS)
        self.assertEqual(clock._read_us(14 * SECOND_NS), 103_000_000)
        clock._restart(7, 5 * SECOND_NS)
        clock.mark_busy(6 * SECOND_NS)
        self.assertEqual(clock._read_us(8 * SECOND_NS), 2_000_007)


class TestPrefillOverlapShutdownVote(CustomTestCase):
    """On shutdown the overlap prefill loop launches nothing new, yet still runs
    the one batch selection whose MLP-sync gather carries a group-local
    ShutdownReq to the other attention-DP ranks."""

    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="scheduler"))

    def _selections_after_shutdown(self, vote_pending: bool) -> int:
        scheduler = TestSchedulerIdleStepCounters.make_scheduler(self, [None])

        def honor_shutdown(**kwargs):
            scheduler.gracefully_exit = True
            return True

        scheduler.ingest_requests = Mock(side_effect=honor_shutdown)
        scheduler.dp_attn_adapter = SimpleNamespace(
            shutdown_vote_pending=lambda: vote_pending
        )
        Scheduler.event_loop_overlap_disagg_prefill(scheduler)
        return scheduler.get_next_disagg_prefill_batch_to_run.call_count

    def test_unsent_vote_still_reaches_the_gather(self):
        self.assertEqual(self._selections_after_shutdown(vote_pending=True), 1)

    def test_no_new_selection_without_a_vote_to_send(self):
        self.assertEqual(self._selections_after_shutdown(vote_pending=False), 0)


if __name__ == "__main__":
    unittest.main()
