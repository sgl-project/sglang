"""A weight update whose cache flush is refused reports failure instead of crashing."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput
from sglang.srt.managers.scheduler_components import weight_updater

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestWeightUpdateFlushFailure(unittest.TestCase):
    def _manager(self, flush_ok: bool):
        runner = MagicMock()
        runner.weight_updater.update_weights_from_disk.return_value = (True, "ok")
        tp_worker = MagicMock()
        tp_worker.weight_update_runners.return_value = [("target", runner)]
        scheduler = MagicMock()
        with patch.object(weight_updater, "get_parallel", MagicMock()):
            return weight_updater.SchedulerWeightUpdaterManager(
                tp_worker=tp_worker,
                draft_worker=None,
                memory_saver_adapter=MagicMock(),
                flush_cache=MagicMock(return_value=flush_ok),
                is_fully_idle=MagicMock(return_value=flush_ok),
                scheduler=scheduler,
            )

    def test_refused_flush_returns_failure(self):
        # A paused scheduler that still has requests refuses the flush.
        manager = self._manager(flush_ok=False)
        out = manager.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(model_path="m", weight_version="v2")
        )
        self.assertFalse(out.success)
        self.assertIn("cache was not flushed", out.message)
        # The served weights did change, so the new version is recorded.
        manager.scheduler.record_weight_version_change.assert_called_once_with(
            new_version="v2"
        )

    def test_successful_flush_unchanged(self):
        manager = self._manager(flush_ok=True)
        out = manager.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(model_path="m", weight_version="v2")
        )
        self.assertTrue(out.success)


if __name__ == "__main__":
    unittest.main()
