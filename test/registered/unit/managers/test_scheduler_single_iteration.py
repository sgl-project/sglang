import unittest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
import torch.distributed
import torch.multiprocessing

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    create_scheduler_watchdog,
)
from sglang.srt.managers.scheduler_components.pause_state import PauseState
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


@dataclass(frozen=True)
class _Batch:
    name: str

    def copy(self):
        return _Batch(self.name)


def _broadcast_local_rank(rank, init_file):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        parallel = SimpleNamespace(
            attn_tp_rank=rank,
            attn_cp_rank=0,
            attn_dp_enabled=False,
            tp_size=2,
            tp_group=SimpleNamespace(
                rank=rank, ranks=[0, 1], cpu_group=torch.distributed.group.WORLD
            ),
        )
        receiver = SchedulerRequestReceiver(
            recv_from_tokenizer=None,
            recv_from_rpc=None,
            recv_skipper=None,
            input_blocker=None,
            mm_receiver=None,
            server_args=None,
            model_config=None,
            max_recv_per_poll=-1,
            request_retirer=None,
            get_last_batch=None,
            rust_server_mode=False,
        )
        module = "sglang.srt.managers.scheduler_components.request_receiver."
        with (
            patch(module + "get_parallel", return_value=parallel),
            patch(module + "get_shm_exchange", return_value=None),
            patch.object(
                SchedulerRequestReceiver,
                "_pull_raw_reqs",
                side_effect=AssertionError("broadcast read external input"),
            ),
        ):
            for expected in ([], ["expired"], []):
                local = [AbortReq(rid=rid) for rid in expected] if rank == 0 else []
                received = receiver.broadcast_local_requests(local)
                assert [req.rid for req in received] == expected
    finally:
        torch.distributed.destroy_process_group()


class TestSchedulerSingleIteration(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.parallel = SimpleNamespace(
            pp_size=1,
            pp_rank=0,
            attn_tp_rank=0,
            attn_cp_rank=0,
            attn_dp_enabled=False,
            tp_size=1,
        )
        for target, value in (
            ("sglang.srt.managers.scheduler.get_parallel", self.parallel),
            (
                "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
                self.parallel,
            ),
            ("sglang.srt.managers.scheduler.is_war_barrier_enabled", False),
            ("sglang.srt.managers.scheduler.use_mlx", False),
        ):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

        # A single step must never reach the tokenizer/RPC input path.
        patcher = patch.object(
            SchedulerRequestReceiver,
            "_pull_raw_reqs",
            side_effect=AssertionError("single step read external input"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _scheduler(self, batches, *, overlap=False, early=None):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._event_loop_driver = None
        scheduler.gracefully_exit = False
        scheduler.pause_state = PauseState()
        scheduler.is_initializing = False
        scheduler.forward_ct = 0
        scheduler.watchdog_heartbeats = 0
        scheduler.cur_batch_for_debug = None
        scheduler.dump_requests_before_crash = Mock()
        scheduler.enable_overlap = overlap
        scheduler.enable_overlap_mlx = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.server_args = SimpleNamespace(enable_pd_role_switch=False)
        scheduler.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[])
        )
        scheduler.device = "cpu"
        scheduler.device_module = SimpleNamespace(StreamContext=lambda _: nullcontext())
        scheduler.schedule_stream = SimpleNamespace()
        scheduler.metrics_reporter = MagicMock()
        scheduler.invariant_checker = MagicMock()
        scheduler.invariant_checker._check_all_pools.return_value = (None, [])
        scheduler.pool_stats_observer = MagicMock()
        scheduler.result_queue = deque()
        scheduler.running_batch = "running"
        scheduler.last_batch = None
        scheduler._sched_idled = False
        scheduler.is_generation = True
        scheduler.enable_unified_memory = False
        scheduler.uocr_runtime = None
        scheduler._poll_timeout_aborts = Mock(return_value=[])
        scheduler.increment_scheduler_iterations = Mock()
        scheduler._record_scheduler_state_for_paused_engine = Mock()
        scheduler.request_receiver = SchedulerRequestReceiver(
            recv_from_tokenizer=Mock(),
            recv_from_rpc=Mock(),
            recv_skipper=None,
            input_blocker=None,
            mm_receiver=None,
            server_args=scheduler.server_args,
            model_config=None,
            max_recv_per_poll=-1,
            request_retirer=Mock(),
            get_last_batch=lambda: scheduler.last_batch,
            rust_server_mode=False,
        )

        events, received, grad_modes = [], [], []
        remaining = iter(batches)
        scheduler.process_input_requests = lambda reqs: received.extend(reqs)

        def select_batch(**_):
            batch = next(remaining)
            events.append(f"select:{batch.name if batch else 'idle'}")
            return SimpleNamespace(running_batch="running", batch_to_run=batch)

        def run_batch(batch):
            scheduler.forward_ct += 1
            events.append(f"run:{batch.name}")
            grad_modes.append(torch.is_grad_enabled())
            return f"result:{batch.name}"

        scheduler.get_next_batch_to_run = select_batch
        scheduler.run_batch = run_batch
        scheduler.process_batch_result = lambda batch, result: events.append(
            f"process:{batch.name}:{result}"
        )
        scheduler.should_process_last_batch_early = lambda batch: (
            batch is not None and batch.name == early
        )
        scheduler._apply_war_barrier = lambda: events.append("barrier")
        scheduler.launch_batch_sample_if_needed = lambda result, batch: events.append(
            f"sample:{batch.name if batch else 'idle'}"
        )
        scheduler.on_idle = lambda: events.append("idle")
        return scheduler, events, received, grad_modes

    def test_normal_steps_once_and_preserves_grad_mode(self):
        scheduler, events, _, grad_modes = self._scheduler([_Batch("a"), None])
        with torch.enable_grad():
            scheduler.run_one_iteration()
            self.assertTrue(torch.is_grad_enabled())
            self.assertEqual(events, ["select:a", "run:a", "process:a:result:a"])
            scheduler.run_one_iteration()

        self.assertEqual(grad_modes, [False])
        self.assertEqual(events[-2:], ["select:idle", "idle"])
        self.assertIsNone(scheduler.last_batch)
        self.assertTrue(scheduler._sched_idled)
        self.assertEqual(scheduler.increment_scheduler_iterations.call_count, 2)
        scheduler.metrics_reporter.start_scheduler_time_accounting.assert_called_once()

    def test_overlap_retains_queue_and_honors_early_processing(self):
        scheduler, events, _, _ = self._scheduler(
            [_Batch("a"), _Batch("b"), None, None], overlap=True, early="a"
        )
        scheduler.run_one_iteration()
        self.assertEqual([batch.name for batch, _ in scheduler.result_queue], ["a"])
        self.assertNotIn("process:a:result:a", events)

        scheduler.run_one_iteration()
        self.assertEqual([batch.name for batch, _ in scheduler.result_queue], ["b"])
        self.assertLess(events.index("process:a:result:a"), events.index("select:b"))

        scheduler.run_one_iteration()
        self.assertEqual(list(scheduler.result_queue), [])
        self.assertIn("process:b:result:b", events)
        scheduler.run_one_iteration()
        self.assertEqual(events[-2:], ["idle", "sample:idle"])
        scheduler.metrics_reporter.start_scheduler_time_accounting.assert_called_once()

    def test_pause_and_shutdown_leave_inflight_result_untouched(self):
        scheduler, events, _, _ = self._scheduler([_Batch("a"), None], overlap=True)
        scheduler.run_one_iteration()
        scheduler.pause_state.pause()
        scheduler.run_one_iteration()
        self.assertEqual([batch.name for batch, _ in scheduler.result_queue], ["a"])
        scheduler._record_scheduler_state_for_paused_engine.assert_called_once()

        scheduler.gracefully_exit = True
        scheduler.run_one_iteration()
        self.assertEqual(scheduler.increment_scheduler_iterations.call_count, 2)
        self.assertNotIn("process:a:result:a", events)

    def test_first_step_preserves_work_on_replaced_schedule_stream(self):
        scheduler, _, _, _ = self._scheduler([None], overlap=True)
        scheduler.device = "cuda"
        original = SimpleNamespace(cuda_stream=1)
        replacement = SimpleNamespace(cuda_stream=2, wait_stream=Mock())
        scheduler.schedule_stream = original
        scheduler.forward_stream = SimpleNamespace(cuda_stream=1)

        with (
            patch("sglang.srt.managers.scheduler.is_cuda", return_value=True),
            patch(
                "sglang.srt.managers.scheduler.allocate_distinct_stream",
                return_value=replacement,
            ),
        ):
            scheduler.run_one_iteration()

        replacement.wait_stream.assert_called_once_with(original)

    @contextmanager
    def _watchdog(self, scheduler, *, soft=False):
        # Drive the real polling loop with a fake clock, without a daemon or signals.
        module = "sglang.srt.utils.watchdog."
        with (
            patch(module + "psutil.Process"),
            patch(module + "pyspy_dump_schedulers"),
        ):
            with patch(module + "threading.Thread"):
                watchdog = create_scheduler_watchdog(scheduler, 10, soft=soft)
            yield watchdog

    def test_watchdog_allows_long_gaps_after_nonempty_steps(self):
        for overlap in (False, True):
            for soft in (False, True):
                with self.subTest(overlap=overlap, soft=soft):
                    scheduler, _, _, _ = self._scheduler([_Batch("a")], overlap=overlap)
                    scheduler.run_one_iteration()
                    with (
                        self._watchdog(scheduler, soft=soft) as watchdog,
                        patch(
                            "sglang.srt.utils.watchdog.time.perf_counter",
                            side_effect=[0, 0, 20, 40],
                        ),
                        patch(
                            "sglang.srt.utils.watchdog.time.sleep",
                            side_effect=[None, None, StopIteration],
                        ),
                    ):
                        self.assertFalse(watchdog.is_active())
                        with self.assertRaises(StopIteration):
                            watchdog._watchdog_once()
                        watchdog.parent_process.send_signal.assert_not_called()
                    self.assertEqual(scheduler.cur_batch_for_debug, _Batch("a"))
                    if overlap:
                        self.assertEqual(len(scheduler.result_queue), 1)

    def test_watchdog_rearms_when_it_misses_the_gap_and_detects_a_stalled_call(self):
        scheduler, _, _, _ = self._scheduler([None])
        scheduler.run_one_iteration()
        scheduler.pause_state.pause()
        entered, release = Event(), Event()

        def hold_input_processing(_requests):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release the scheduler")

        scheduler.process_input_requests = hold_input_processing
        now = 0
        polls = 0
        with self._watchdog(scheduler) as watchdog, ThreadPoolExecutor(1) as executor:
            step = executor.submit(scheduler.run_one_iteration)

            def advance_clock(_seconds):
                nonlocal now, polls, step
                polls += 1
                if polls == 1:
                    release.set()
                    step.result(timeout=10)
                    entered.clear()
                    release.clear()
                    # No watchdog poll sees the inactive interval. No forward
                    # occurs in either paused step to refresh its counter.
                    now = 100
                    step = executor.submit(scheduler.run_one_iteration)
                    self.assertTrue(entered.wait(10))
                elif polls == 2:
                    now = 105
                elif polls == 3:
                    now = 111
                elif polls > 4:
                    self.fail("watchdog did not time out the stalled call")

            try:
                self.assertTrue(entered.wait(10))
                with (
                    patch(
                        "sglang.srt.utils.watchdog.time.perf_counter",
                        side_effect=lambda: now,
                    ),
                    patch(
                        "sglang.srt.utils.watchdog.time.sleep",
                        side_effect=advance_clock,
                    ),
                ):
                    watchdog._watchdog_once()
                self.assertEqual(now, 111)
                watchdog.parent_process.send_signal.assert_called_once()
                scheduler.dump_requests_before_crash.assert_called_once()
            finally:
                release.set()
                step.result(timeout=10)
            self.assertFalse(watchdog.is_active())

    def test_watchdog_disarms_on_exception_and_rearms_before_input_processing(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                scheduler, _, _, _ = self._scheduler([], overlap=overlap)
                with self._watchdog(scheduler) as watchdog:
                    previous_counter = watchdog.get_counter()

                    def fail_input_processing(_requests):
                        self.assertTrue(watchdog.is_active())
                        self.assertNotEqual(watchdog.get_counter(), previous_counter)
                        raise ValueError("input failed")

                    scheduler.process_input_requests = fail_input_processing
                    for _ in range(2):
                        with self.assertRaisesRegex(ValueError, "input failed"):
                            scheduler.run_one_iteration()
                        self.assertFalse(watchdog.is_active())
                        previous_counter = watchdog.get_counter()

    def test_watchdog_preserves_initialization_and_continuous_loop_activity(self):
        scheduler, _, _, _ = self._scheduler([_Batch("a")])
        scheduler.run_one_iteration()
        with self._watchdog(scheduler) as watchdog:
            scheduler._event_loop_driver = "continuous"
            self.assertTrue(watchdog.is_active())
            scheduler.cur_batch_for_debug = None
            self.assertFalse(watchdog.is_active())
            scheduler.is_initializing = True
            self.assertTrue(watchdog.is_active())

    def test_timeout_abort_reaches_every_tp_rank_without_socket_input(self):
        first, _, first_received, _ = self._scheduler([None])
        second, _, second_received, _ = self._scheduler([None])
        abort = AbortReq(rid="expired")
        first._poll_timeout_aborts.return_value = [abort]
        self.parallel.tp_size = 2
        self.parallel.tp_group = SimpleNamespace(
            rank=0, cpu_group=object(), ranks=[0, 1]
        )
        broadcast = []

        def distribute(reqs, rank, _group, src):
            self.assertEqual(src, 0)
            if rank == 0:
                broadcast.extend(reqs)
            return list(broadcast)

        with (
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.get_shm_exchange",
                return_value=None,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.broadcast_pyobj",
                side_effect=distribute,
            ),
        ):
            first.run_one_iteration()
            self.parallel.attn_tp_rank = 1
            self.parallel.tp_group.rank = 1
            second.run_one_iteration()

        self.assertEqual(first_received, [abort])
        self.assertEqual(second_received, [abort])
        second._poll_timeout_aborts.assert_not_called()

    def test_timeout_broadcast_across_two_gloo_ranks(self):
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _broadcast_local_rank,
                args=(str(Path(directory) / "gloo-init"),),
                nprocs=2,
                join=True,
            )

    def test_requests_can_be_dispatched_before_the_first_step(self):
        scheduler, events, received, _ = self._scheduler([_Batch("a")])
        scheduler.scheduler_stage_metrics = None
        scheduler.session_controller = MagicMock()
        scheduler.flush_wrapper = MagicMock()
        scheduler.external_corpus_manager = None
        scheduler._tokenized_requests = lambda _: []
        scheduler._request_dispatcher = lambda req: received.append(req)
        scheduler.process_input_requests = Scheduler.process_input_requests.__get__(
            scheduler
        )
        request = AbortReq(rid="injected")
        with (
            patch(
                "sglang.srt.managers.scheduler.get_mm",
                return_value=SimpleNamespace(mm_feature_transport="none"),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_disagg",
                return_value=SimpleNamespace(enable_pd_role_switch=False),
            ),
        ):
            scheduler.process_input_requests([request])
            scheduler.run_one_iteration()

        self.assertEqual(received, [request])
        self.assertIn("process:a:result:a", events)

    def test_continuous_loops_still_receive_external_requests(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                scheduler, events, received, _ = self._scheduler(
                    [_Batch("a")], overlap=overlap
                )
                abort = AbortReq(rid="external")
                with patch.object(
                    SchedulerRequestReceiver,
                    "recv_requests",
                    side_effect=[[abort], StopIteration],
                ):
                    with self.assertRaises(StopIteration):
                        if overlap:
                            scheduler.event_loop_overlap()
                        else:
                            scheduler.event_loop_normal()
                self.assertEqual(received, [abort])
                self.assertIn("run:a", events)

    def test_unsupported_modes_and_mixed_drivers_are_rejected(self):
        for mode in ("pp", "pd", "role_switch", "mlx"):
            with self.subTest(mode=mode):
                scheduler, _, _, _ = self._scheduler([None])
                self.parallel.pp_size = 2 if mode == "pp" else 1
                if mode == "pd":
                    scheduler.disaggregation_mode = DisaggregationMode.PREFILL
                elif mode == "role_switch":
                    scheduler.server_args.enable_pd_role_switch = True
                elif mode == "mlx":
                    scheduler.enable_overlap_mlx = True
                with self.assertRaises(NotImplementedError):
                    scheduler.run_one_iteration()
        self.parallel.pp_size = 1
        scheduler, _, _, _ = self._scheduler([None])
        scheduler._event_loop_driver = "continuous"
        with self.assertRaises(RuntimeError):
            scheduler.run_one_iteration()
        scheduler._event_loop_driver = "single_iteration"
        with self.assertRaises(RuntimeError):
            scheduler.run_event_loop()


if __name__ == "__main__":
    unittest.main()
