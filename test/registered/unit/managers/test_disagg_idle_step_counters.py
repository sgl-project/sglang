"""Idle gaps must not inflate prefill busy time or decode timing samples."""

import importlib.util
import sys
import unittest
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from parameterized import parameterized

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

LAUNCH_TIMESTAMPS = (0.0, 0.125, 1.0, 1.125)
PP_MODULE = "sglang.srt.managers.scheduler_pp_mixin"
PDMUX_MODULE = "sglang.srt.multiplex.multiplexing_mixin"


class _BeforeModelForward(Exception):
    pass


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


class TestSchedulerIdleStepCounters(CustomTestCase):
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
                scheduler = self.make_scheduler(schedule)
                self.run_and_check(scheduler, event_loop, mode, idle_iterations > 0)

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
                        self.prepare_pp_scheduler(scheduler)
                        if pending_transfer:
                            if name == "prefill":
                                scheduler.disagg_prefill_inflight_queue = [object()]
                            else:
                                scheduler.disagg_decode_transfer_queue.queue = [
                                    object()
                                ]
                        parallel = SimpleNamespace(
                            pp_async_batch_depth=depth,
                            enable_dsa_prefill_context_parallel=False,
                        )
                        with (
                            patch(f"{PP_MODULE}.get_parallel", return_value=parallel),
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
                        # Transfers suppress housekeeping, not the idle flag.
                        self.assertEqual(
                            scheduler.on_idle.call_count,
                            0 if pending_transfer else 1 + (pattern == "idle_cycle"),
                        )

    def test_pdmux_idle_gap(self):
        for idle_iterations in (0, 1, 2):
            with self.subTest(idle_iterations=idle_iterations):
                batches = self.make_batches(ForwardMode.DECODE)
                schedule = batches[:2] + [None] * idle_iterations + batches[2:] + [None]
                scheduler = self.make_scheduler(schedule)
                scheduler.stream_groups = [(Mock(), Mock())]
                scheduler.sm_counts = [(80, 52)]
                scheduler.split_prefill_batch = None
                scheduler.update_split_prefill_batch = lambda sm_count, running_batch: (
                    False,
                    running_batch,
                )
                scheduler.update_running_batch = Mock(
                    side_effect=[batch or ScheduleBatch(reqs=[]) for batch in schedule]
                )
                with (
                    patch(f"{PDMUX_MODULE}.get_current_stream_idx", return_value=0),
                    patch(f"{PDMUX_MODULE}.set_pdmux_status"),
                    patch(f"{PDMUX_MODULE}.torch.cuda.empty_cache"),
                    patch(
                        f"{PDMUX_MODULE}.torch.cuda.stream",
                        side_effect=lambda stream: nullcontext(),
                    ),
                ):
                    self.run_and_check(
                        scheduler,
                        Scheduler.event_loop_pdmux,
                        ForwardMode.DECODE,
                        idle_iterations > 0,
                    )

    @parameterized.expand([("fresh", False), ("chained", True)])
    def test_mlx_idle_gap(self, name, chained):
        module = load_mlx_scheduler_module()
        mixin = module.SchedulerMlxOverlapMixin
        mode = ForwardMode.DECODE if chained else ForwardMode.EXTEND
        for idle_iterations in (0, 1, 2):
            with self.subTest(idle_iterations=idle_iterations):
                batches = self.make_batches(mode)
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
                scheduler.gracefully_exit = False
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
                scheduler.profiler_manager = SimpleNamespace(
                    _profile_batch_predicate=Mock()
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
                        module.time, "monotonic", side_effect=LAUNCH_TIMESTAMPS
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

    def make_batches(self, mode):
        return [
            ScheduleBatch(
                reqs=[
                    SimpleNamespace(rid="request", finished=Mock(return_value=False))
                ],
                forward_mode=mode,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                launch_ts=launch_ts,
                extend_num_tokens=1024,
            )
            for launch_ts in LAUNCH_TIMESTAMPS
        ]

    def run_and_check(self, scheduler, event_loop, mode, after_idle):
        observed_idle_flags = []
        observed_iters = []

        def run_batch(batch, pp_proxy_tensors=None):
            # Exercise the real timestamp, iteration, and flag handoff. Only
            # model execution is stopped, at the scripted pre-forward hook.
            with patch(
                "sglang.srt.managers.scheduler.time.monotonic",
                return_value=batch.launch_ts,
            ):
                with self.assertRaises(_BeforeModelForward):
                    Scheduler.run_batch(scheduler, batch, pp_proxy_tensors)
            return GenerationBatchResult()

        def process_batch_result(batch, result):
            observed_idle_flags.append(batch.after_idle_gap)
            observed_iters.append(batch.forward_iter)
            scheduler._record_step_counters(batch, result)

        scheduler.run_batch = run_batch
        scheduler.process_batch_result = process_batch_result
        with self.assertRaises(StopIteration):
            event_loop(scheduler)

        self.assertEqual(observed_idle_flags, [False, False, after_idle, False])
        self.assertEqual(scheduler.forward_ct, 4)
        self.assertEqual(observed_iters, [1, 2, 3, 4])
        expected_intervals = [
            LAUNCH_TIMESTAMPS[1] - LAUNCH_TIMESTAMPS[0],
            LAUNCH_TIMESTAMPS[3] - LAUNCH_TIMESTAMPS[2],
        ]
        if not after_idle:
            expected_intervals.append(LAUNCH_TIMESTAMPS[2] - LAUNCH_TIMESTAMPS[1])
        expected_samples = len(expected_intervals)
        expected_busy_us = round(sum(expected_intervals) * 1_000_000)
        if mode == ForwardMode.EXTEND:
            self.assertEqual(scheduler.total_prefill_busy_us, expected_busy_us)
            self.assertEqual(
                scheduler.total_prefill_uncached_tokens, expected_samples * 1024
            )
        else:
            self.assertEqual(scheduler.decode_moment_totals[0], expected_samples)
            self.assertEqual(scheduler.decode_moment_totals[2], expected_busy_us)

    def make_scheduler(self, schedule):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._engine_paused = False
        scheduler._sched_idled = False
        scheduler._prev_step = None
        scheduler.forward_ct = 0
        scheduler.processed_tokens_counter = 0
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.ps = SimpleNamespace(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0)
        scheduler._poll_timeout_aborts = Mock(return_value=[])
        scheduler.scheduler_stage_metrics = None
        scheduler.metrics_reporter = SimpleNamespace(record_scheduler_active=Mock())
        scheduler.scripted_scheduler_hook = SimpleNamespace(
            on_run_batch=Mock(side_effect=_BeforeModelForward)
        )
        scheduler.total_prefill_busy_us = 0
        scheduler.total_prefill_uncached_tokens = 0
        scheduler.decode_moment_totals = [0.0] * 6
        scheduler.running_batch = ScheduleBatch(reqs=[])
        scheduler.last_batch = None
        scheduler.chunked_req = None
        scheduler.waiting_queue = []
        scheduler.enable_staging = False
        scheduler.request_receiver = SimpleNamespace(
            recv_requests=Mock(side_effect=[[]] * len(schedule) + [StopIteration])
        )
        scheduler.process_input_requests = Mock()
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            pop_bootstrapped=Mock(return_value=[])
        )
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            prefetch_prefill_dp_rank_queries=Mock(), queue=[]
        )
        scheduler.process_decode_queue = Mock()
        scheduler.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, chunked_req: batch
        )
        next_plan = Mock(
            side_effect=[
                SimpleNamespace(
                    running_batch=scheduler.running_batch, batch_to_run=batch
                )
                for batch in schedule
            ]
        )
        scheduler.get_next_disagg_prefill_batch_to_run = next_plan
        scheduler.get_next_disagg_decode_batch_to_run = next_plan
        scheduler.get_next_batch_to_run = next_plan
        scheduler.get_new_batch_prefill = next_plan
        scheduler.is_disable_overlap_for_batch = Mock(return_value=False)
        scheduler._apply_war_barrier = Mock()
        scheduler.process_disagg_prefill_inflight_queue = Mock()
        scheduler.launch_batch_sample_if_needed = Mock()
        scheduler.on_idle = Mock()
        return scheduler

    def prepare_pp_scheduler(self, scheduler):
        get_parallel().pp_size = 2
        scheduler.pp_group = SimpleNamespace(is_last_rank=True)
        scheduler.forward_stream_ctx = nullcontext()
        scheduler.forward_stream = Mock()
        scheduler.schedule_stream = Mock()
        scheduler.device_module = SimpleNamespace(Event=Mock, current_stream=Mock())
        scheduler._pp_recv_proxy_tensors = Mock()
        scheduler._pp_commit_comm_work = Mock()
        scheduler._pp_prepare_tensor_dict = Mock(return_value={})
        scheduler._pp_commit_send_output_work_and_preprocess_output_tensors = Mock(
            return_value=(None, GenerationBatchResult(), Mock())
        )
        scheduler._pp_pd_get_bootstrapped_ids = Mock(return_value=None)
        scheduler._pp_pd_get_prefill_transferred_ids = Mock(return_value=None)
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
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])


if __name__ == "__main__":
    unittest.main()
