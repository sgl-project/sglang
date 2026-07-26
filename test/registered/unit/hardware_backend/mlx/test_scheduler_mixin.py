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

import dataclasses
import importlib.util
import platform
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
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


class _FakeReq:
    def finished(self):
        return False


def _make_fake_schedule_batch(forward_mode) -> ScheduleBatch:
    # Auto-fill every List/Tensor field so ScheduleBatch.copy() (which reads
    # ~30 fields) doesn't hit an unset attribute; unlisted fields keep their
    # dataclass default.
    batch = ScheduleBatch(reqs=[_FakeReq()])
    for field in dataclasses.fields(ScheduleBatch):
        annotation = str(field.type)
        if "List" in annotation or "list[" in annotation:
            setattr(batch, field.name, [])
        elif "Tensor" in annotation:
            setattr(batch, field.name, torch.zeros(1, dtype=torch.int64))
    batch.forward_mode = forward_mode
    # Make resolve_forward_inputs() a no-op: skip both its H2D-copy and
    # FutureMap-relay branches.
    batch.prefill_input_ids_cpu = None
    batch.input_ids = torch.zeros(1, dtype=torch.int64)
    batch.enable_overlap = False
    return batch


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestEventLoopOverlapMlxSetsLaunchTs(unittest.TestCase):
    """event_loop_overlap_mlx bypasses run_batch(), which is where launch_ts is
    normally stamped, so _launch_fresh/_launch_chained must set it directly.
    Without that, _record_step_counters crashes on time.monotonic() - None for
    the very first batch (see the scheduler.py call it feeds).
    """

    class _StopLoop(Exception):
        pass

    def _make_scheduler(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = MagicMock()
        scheduler.waiting_queue = []
        scheduler.running_batch = None
        scheduler.last_batch = None
        scheduler._engine_paused = False
        scheduler.result_queue = deque()
        scheduler.tp_worker.finalize_mlx_result.return_value = MagicMock(
            next_token_ids=None
        )
        # Route the loop's self._finalize_mlx_pending_job(...) call to the
        # real mixin method (bound to this mock) instead of an auto-mock, so
        # process_batch_result actually gets invoked with the launched batch.
        scheduler._finalize_mlx_pending_job = (
            lambda pending: SchedulerMlxOverlapMixin._finalize_mlx_pending_job(
                scheduler, pending
            )
        )
        return scheduler, SchedulerMlxOverlapMixin

    def test_fresh_launch_sets_launch_ts_before_process_batch_result(self):
        scheduler, mixin_cls = self._make_scheduler()
        fresh_batch = _make_fake_schedule_batch(ForwardMode.EXTEND)

        scheduler.get_next_batch_to_run.side_effect = [
            MagicMock(batch_to_run=fresh_batch, running_batch=None),
            MagicMock(batch_to_run=None, running_batch=None),
        ]
        scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = (
            MagicMock(),  # lazy_tokens
            [],  # prefills
            [],  # extends
            None,  # decode
            "extend",  # mode
        )
        scheduler.request_receiver.recv_requests.side_effect = [
            [],
            [],
            self._StopLoop(),
        ]

        with self.assertRaises(self._StopLoop):
            mixin_cls.event_loop_overlap_mlx(scheduler)

        scheduler.process_batch_result.assert_called_once()
        finalized_batch = scheduler.process_batch_result.call_args[0][0]
        self.assertIsNotNone(
            finalized_batch.launch_ts,
            "_launch_fresh must set launch_ts; the MLX overlap loop never "
            "calls run_batch(), which is where it's normally set.",
        )

    def test_chained_launch_gets_its_own_launch_ts(self):
        scheduler, mixin_cls = self._make_scheduler()
        fresh_batch = _make_fake_schedule_batch(ForwardMode.DECODE)

        scheduler.get_next_batch_to_run.return_value = MagicMock(
            batch_to_run=fresh_batch, running_batch=None
        )
        scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = (
            MagicMock(),
            [],
            [],
            MagicMock(),  # decode: non-None so chaining is eligible
            "decode",
        )
        scheduler.tp_worker.async_chained_decode_mlx.return_value = (
            MagicMock(),
            [],
            [],
            MagicMock(),
            "decode",
        )
        scheduler.request_receiver.recv_requests.side_effect = [
            [],
            [],
            self._StopLoop(),
        ]

        with patch("time.monotonic", side_effect=[100.0, 200.0, 300.0, 400.0]):
            with self.assertRaises(self._StopLoop):
                mixin_cls.event_loop_overlap_mlx(scheduler)

        scheduler.process_batch_result.assert_called_once()
        fresh_launch_ts = scheduler.process_batch_result.call_args[0][0].launch_ts
        chained_job = scheduler.result_queue.pop()
        self.assertIsNotNone(chained_job.batch_copy.launch_ts)
        self.assertNotEqual(
            chained_job.batch_copy.launch_ts,
            fresh_launch_ts,
            "a chained step is a real, separate launch and must get its own "
            "launch_ts rather than inheriting prev's, or the decode-step "
            "interval _record_step_counters computes collapses to zero.",
        )


if __name__ == "__main__":
    unittest.main()
