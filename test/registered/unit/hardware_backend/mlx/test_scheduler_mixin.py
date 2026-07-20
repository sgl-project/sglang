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

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
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


if __name__ == "__main__":
    unittest.main()
