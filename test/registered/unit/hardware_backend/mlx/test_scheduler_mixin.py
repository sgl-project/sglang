"""Unit tests for the MLX overlap scheduler mixin (hardware_backend/mlx/scheduler_mixin.py).

Covers:
  - Every MLX launch advances forward_ct and stamps forward_iter/launch_ts.
  - The profiler predicate runs before the async forward is enqueued, matching
    Scheduler.run_batch() so step-bounded profiling stops on the right step.

Skips on non-Apple-Silicon platforms and when ``mlx`` is missing (importing
scheduler_mixin requires ``mlx.core``).
"""

from __future__ import annotations

import importlib.util
import platform
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires Apple Silicon and mlx"


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMlxLaunchBookkeeping(unittest.TestCase):
    """run_batch-style bookkeeping for the MLX overlap loop."""

    def _make_scheduler(self):
        scheduler = MagicMock()
        scheduler.forward_ct = 0
        result = MagicMock()
        result.next_token_ids = None
        scheduler.tp_worker.finalize_mlx_result.return_value = result
        return scheduler

    def test_prepare_launch_advances_forward_ct_and_runs_predicate(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = self._make_scheduler()
        batch = MagicMock()

        SchedulerMlxOverlapMixin._prepare_mlx_launch(scheduler, batch)

        self.assertEqual(scheduler.forward_ct, 1)
        self.assertEqual(batch.forward_iter, 1)
        self.assertIsInstance(batch.launch_ts, float)
        scheduler.profiler_manager._profile_batch_predicate.assert_called_once_with(
            batch
        )

    def test_forward_ct_advances_once_per_launch(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = self._make_scheduler()

        for expected in (1, 2, 3):
            SchedulerMlxOverlapMixin._prepare_mlx_launch(scheduler, MagicMock())
            self.assertEqual(scheduler.forward_ct, expected)

        self.assertEqual(
            scheduler.profiler_manager._profile_batch_predicate.call_count, 3
        )

    def test_finalize_does_not_double_count_launch(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = self._make_scheduler()
        pending = MagicMock()

        SchedulerMlxOverlapMixin._prepare_mlx_launch(scheduler, pending.batch_copy)
        SchedulerMlxOverlapMixin._finalize_mlx_pending_job(scheduler, pending)

        self.assertEqual(scheduler.forward_ct, 1)
        self.assertEqual(pending.batch_copy.forward_iter, 1)
        scheduler.process_batch_result.assert_called_once()


class _StopLoop(Exception):
    """Sentinel to break out of the event loop's ``while True``."""


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestOverlapLoopStampsLaunchTs(unittest.TestCase):
    """Every batch the MLX overlap loop launches must carry ``launch_ts``.

    ``Scheduler.run_batch`` stamps ``batch.launch_ts`` on every forward, and
    ``process_batch_result`` -> ``_record_step_counters`` subtracts it
    unconditionally for prefill/decode batches.  The MLX overlap loop bypasses
    ``run_batch``, so if its launch paths skip the stamp, the first real
    request's result processing raises ``TypeError: float - NoneType`` and
    kills the scheduler (health-check requests are filtered from the counters,
    which keeps ``/health_generate`` green while every real request crashes).
    """

    def _make_scheduler(self, *, recv_side_effect):
        from collections import deque

        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = MagicMock()
        scheduler.forward_ct = 0
        scheduler._prepare_mlx_launch.side_effect = lambda batch: (
            SchedulerMlxOverlapMixin._prepare_mlx_launch(scheduler, batch)
        )
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler.waiting_queue = []
        scheduler.result_queue = deque()
        scheduler.request_receiver.recv_requests.side_effect = recv_side_effect
        result = MagicMock()
        result.next_token_ids = None
        scheduler.tp_worker.finalize_mlx_result.return_value = result
        return scheduler

    def test_fresh_launch_stamps_launch_ts_before_input_resolution(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxLaunch

        scheduler = self._make_scheduler(recv_side_effect=[[], _StopLoop()])

        batch = MagicMock()
        events = []
        scheduler.profiler_manager._profile_batch_predicate.side_effect = (
            lambda _batch: events.append("profile")
        )
        launch_ts_at_copy_time = []
        batch.copy.side_effect = lambda: (
            launch_ts_at_copy_time.append(batch.launch_ts),
            MagicMock(),
        )[1]
        plan = MagicMock()
        plan.batch_to_run = batch
        scheduler.get_next_batch_to_run.return_value = plan
        scheduler.tp_worker.async_forward_batch_generation_mlx.side_effect = (
            lambda _batch: (
                events.append("forward"),
                MlxLaunch(
                    lazy_tokens=None,
                    prefills=[],
                    extends=[],
                    decode=None,
                    mode="extend",
                ),
            )[1]
        )

        with (
            patch(
                "sglang.srt.hardware_backend.mlx.scheduler_mixin.time.monotonic",
                side_effect=lambda: (events.append("launch_ts"), 1.0)[1],
            ),
            patch(
                "sglang.srt.hardware_backend.mlx.scheduler_mixin.resolve_forward_inputs",
                side_effect=lambda *_args: events.append("resolve_inputs"),
            ),
            self.assertRaises(_StopLoop),
        ):
            SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        self.assertEqual(events, ["launch_ts", "profile", "resolve_inputs", "forward"])
        self.assertEqual(len(launch_ts_at_copy_time), 1)
        self.assertEqual(launch_ts_at_copy_time[0], 1.0)

    def test_chained_launch_restamps_launch_ts(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxLaunch

        # Iteration 1: fresh decode launch.  Iteration 2: chain a second
        # decode on top of it.  Iteration 3: stop.
        scheduler = self._make_scheduler(recv_side_effect=[[], [], _StopLoop()])

        events = []
        req = MagicMock()
        req.finished.return_value = False
        batch = MagicMock()
        batch.reqs = [req]
        fresh_copy = MagicMock()
        batch.copy.return_value = fresh_copy
        chained_copy = MagicMock()
        chained_copy.launch_ts = None
        fresh_copy.copy.return_value = chained_copy
        plan = MagicMock()
        plan.batch_to_run = batch
        scheduler.get_next_batch_to_run.return_value = plan

        pending_decode = MagicMock()
        scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = MlxLaunch(
            lazy_tokens=MagicMock(),
            prefills=[],
            extends=[],
            decode=pending_decode,
            mode="decode",
        )
        scheduler.tp_worker.async_chained_decode_mlx.side_effect = lambda _decode: (
            events.append("chained_forward"),
            MlxLaunch(
                lazy_tokens=MagicMock(),
                prefills=[],
                extends=[],
                decode=MagicMock(),
                mode="decode",
            ),
        )[1]

        launch_times = iter((1.0, 2.0))

        def record_launch_ts():
            launch_ts = next(launch_times)
            events.append(f"launch_ts:{launch_ts}")
            return launch_ts

        with (
            patch(
                "sglang.srt.hardware_backend.mlx.scheduler_mixin.time.monotonic",
                side_effect=record_launch_ts,
            ),
            patch(
                "sglang.srt.hardware_backend.mlx.scheduler_mixin.resolve_forward_inputs"
            ),
            self.assertRaises(_StopLoop),
        ):
            SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        scheduler.tp_worker.async_chained_decode_mlx.assert_called_once()
        self.assertLess(events.index("launch_ts:2.0"), events.index("chained_forward"))
        self.assertEqual(chained_copy.launch_ts, 2.0)
        # The live batch only needs the iteration for SWA maintenance before
        # the next fresh launch; per-step timing consumes the batch copy.
        self.assertEqual(batch.forward_iter, 2)
        self.assertEqual(batch.launch_ts, 1.0)


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestOverlapLoopGracefulExit(unittest.TestCase):
    """The MLX overlap loop must honor ``gracefully_exit`` like the standard loops.

    ``handle_shutdown`` (ShutdownReq) only sets ``scheduler.gracefully_exit``;
    actual teardown happens after the event loop returns —
    ``run_scheduler_process``'s ``finally`` calls ``release_host_resources()``
    only once the loop breaks.  ``event_loop_normal`` and ``event_loop_overlap``
    check the flag at the top of every iteration; a loop that never checks it
    spins forever, so the TokenizerManager's shutdown path times out after its
    15 s grace period and falls back to ``kill_process_tree`` — host resources
    never get their user-space release.
    """

    def _make_scheduler(self, *, recv_side_effect):
        from collections import deque

        scheduler = MagicMock()
        scheduler.forward_ct = 0
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler.waiting_queue = []
        scheduler.result_queue = deque()
        scheduler.request_receiver.recv_requests.side_effect = recv_side_effect
        # Model handle_shutdown: processing a non-empty recv batch (the
        # ShutdownReq) flips the flag; the loop must notice at the top of the
        # next iteration instead of polling forever.
        scheduler.process_input_requests.side_effect = lambda reqs: (
            setattr(scheduler, "gracefully_exit", True) if reqs else None
        )
        plan = MagicMock()
        plan.batch_to_run = None
        scheduler.get_next_batch_to_run.return_value = plan
        return scheduler

    def test_loop_exits_after_shutdown_req(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        # Iteration 1: recv the ShutdownReq stand-in (flag flips inside
        # process_input_requests).  Iteration 2 must break before polling
        # again; the sentinel raising instead means the loop never exits.
        scheduler = self._make_scheduler(recv_side_effect=[[MagicMock()], _StopLoop()])

        with patch(
            "sglang.srt.hardware_backend.mlx.scheduler_mixin.mx.synchronize"
        ) as synchronize:
            SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        self.assertEqual(scheduler.request_receiver.recv_requests.call_count, 1)
        synchronize.assert_called_once_with()

    def test_loop_exits_when_shutdown_arrives_while_paused(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        # A paused engine still recvs and processes control requests — that is
        # how unpause (and shutdown) arrive — but `continue`s past the rest of
        # the body.  The flag check must sit above the paused-continue, like in
        # event_loop_normal/event_loop_overlap, or shutdown during a pause
        # spins forever.
        scheduler = self._make_scheduler(recv_side_effect=[[MagicMock()], _StopLoop()])
        scheduler._engine_paused = True

        with patch(
            "sglang.srt.hardware_backend.mlx.scheduler_mixin.mx.synchronize"
        ) as synchronize:
            SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        self.assertEqual(scheduler.request_receiver.recv_requests.call_count, 1)
        scheduler.get_next_batch_to_run.assert_not_called()
        synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
