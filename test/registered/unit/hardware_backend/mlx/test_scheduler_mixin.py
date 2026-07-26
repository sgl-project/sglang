"""Unit tests for the MLX overlap scheduler mixin (hardware_backend/mlx/scheduler_mixin.py).

Covers:
  - _finalize_mlx_pending_job advances forward_ct once per completed step
  - _finalize_mlx_pending_job calls the profiler batch predicate with the
    finalized batch, so step-bounded profiling (``--profile-steps`` /
    ``/start_profile`` num_steps) can auto-stop on the MLX overlap loop, which
    bypasses the standard Scheduler.run_batch().

Skips on non-Apple-Silicon platforms and when ``mlx`` is missing (importing
scheduler_mixin requires ``mlx.core``).
"""

from __future__ import annotations

import importlib.util
import platform
import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires Apple Silicon and mlx"


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestFinalizeMlxPendingJob(unittest.TestCase):
    """forward_ct accounting + profiler predicate wiring in the overlap loop."""

    def _make_scheduler(self):
        scheduler = MagicMock()
        scheduler.forward_ct = 0
        result = MagicMock()
        result.next_token_ids = None
        scheduler.tp_worker.finalize_mlx_result.return_value = result
        return scheduler

    def test_finalize_advances_forward_ct_and_runs_predicate(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = self._make_scheduler()
        pending = MagicMock()

        SchedulerMlxOverlapMixin._finalize_mlx_pending_job(scheduler, pending)

        # Standard run_batch() advances forward_ct and runs the profiler
        # predicate; the MLX overlap loop must do the same here.
        self.assertEqual(scheduler.forward_ct, 1)
        scheduler.profiler_manager._profile_batch_predicate.assert_called_once_with(
            pending.schedule_batch
        )
        # The rest of finalization still runs.
        scheduler.process_batch_result.assert_called_once()

    def test_forward_ct_advances_once_per_step(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = self._make_scheduler()

        for expected in (1, 2, 3):
            SchedulerMlxOverlapMixin._finalize_mlx_pending_job(scheduler, MagicMock())
            self.assertEqual(scheduler.forward_ct, expected)

        self.assertEqual(
            scheduler.profiler_manager._profile_batch_predicate.call_count, 3
        )

    def test_finalize_stamps_forward_iter_like_run_batch(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = self._make_scheduler()
        pending = MagicMock()

        SchedulerMlxOverlapMixin._finalize_mlx_pending_job(scheduler, pending)

        # run_batch assigns batch.forward_iter = forward_ct; the metrics
        # reporter and SWA maintenance skip batches where it is None, so the
        # MLX loop must stamp it on the batch handed to process_batch_result.
        self.assertEqual(pending.batch_copy.forward_iter, 1)
        self.assertEqual(pending.schedule_batch.forward_iter, 1)


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

        scheduler = MagicMock()
        scheduler.forward_ct = 0
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler.waiting_queue = []
        scheduler.result_queue = deque()
        scheduler.request_receiver.recv_requests.side_effect = recv_side_effect
        result = MagicMock()
        result.next_token_ids = None
        scheduler.tp_worker.finalize_mlx_result.return_value = result
        return scheduler

    def test_fresh_launch_stamps_launch_ts_before_copy(self):
        from unittest.mock import patch

        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = self._make_scheduler(recv_side_effect=[[], _StopLoop()])

        batch = MagicMock()
        launch_ts_at_copy_time = []
        batch.copy.side_effect = lambda: (
            launch_ts_at_copy_time.append(batch.launch_ts),
            MagicMock(),
        )[1]
        plan = MagicMock()
        plan.batch_to_run = batch
        scheduler.get_next_batch_to_run.return_value = plan
        scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = (
            None,
            [],
            [],
            None,
            "extend",
        )

        with patch(
            "sglang.srt.hardware_backend.mlx.scheduler_mixin.resolve_forward_inputs"
        ):
            with self.assertRaises(_StopLoop):
                SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        self.assertEqual(len(launch_ts_at_copy_time), 1)
        self.assertIsInstance(
            launch_ts_at_copy_time[0],
            float,
            msg=(
                "MLX overlap loop launched a batch without stamping "
                "launch_ts before batch.copy(); process_batch_result -> "
                "_record_step_counters will crash on float - None."
            ),
        )

    def test_chained_launch_restamps_launch_ts(self):
        from unittest.mock import patch

        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        # Iteration 1: fresh decode launch.  Iteration 2: chain a second
        # decode on top of it.  Iteration 3: stop.
        scheduler = self._make_scheduler(recv_side_effect=[[], [], _StopLoop()])

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
        scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = (
            MagicMock(),
            [],
            [],
            pending_decode,
            "decode",
        )
        scheduler.tp_worker.async_chained_decode_mlx.return_value = (
            MagicMock(),
            [],
            [],
            MagicMock(),
            "decode",
        )

        with patch(
            "sglang.srt.hardware_backend.mlx.scheduler_mixin.resolve_forward_inputs"
        ):
            with self.assertRaises(_StopLoop):
                SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        scheduler.tp_worker.async_chained_decode_mlx.assert_called_once()
        self.assertIsInstance(
            chained_copy.launch_ts,
            float,
            msg=(
                "Chained decode launches must re-stamp launch_ts on their own "
                "batch copy; inheriting the previous step's stamp skews "
                "_record_step_counters' decode inter-step timing, and a None "
                "stamp crashes result processing."
            ),
        )


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

        SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        self.assertEqual(scheduler.request_receiver.recv_requests.call_count, 1)

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

        SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        self.assertEqual(scheduler.request_receiver.recv_requests.call_count, 1)
        scheduler.get_next_batch_to_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
