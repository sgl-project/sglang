import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSchedulerProfilerManager(CustomTestCase):
    def _make_memory_manager(self, output_dir):
        manager = SchedulerProfilerManager(
            ps=SimpleNamespace(tp_rank=0, dp_size=1, pp_size=1, moe_ep_size=1),
            dp_tp_cpu_group=None,
            get_forward_ct=lambda: 0,
        )
        manager.profile_in_progress = True
        manager.torch_profiler_output_dir = Path(output_dir)
        manager.profiler_activities = ["MEM"]
        manager.profile_id = "memory-profile"
        manager.profile_prefix = ""
        manager.merge_profiles = False
        manager.profiler_start_forward_ct = 1
        return manager

    @patch(
        "sglang.srt.managers.scheduler_components.profiler_manager.set_detailed_annotations_enabled"
    )
    @patch(
        "sglang.srt.managers.scheduler_components.profiler_manager.torch.cuda.memory._record_memory_history"
    )
    @patch(
        "sglang.srt.managers.scheduler_components.profiler_manager.torch.cuda.memory._dump_snapshot",
        side_effect=RuntimeError("stoi"),
    )
    def test_memory_snapshot_failure_returns_failure_and_cleans_up(
        self, mock_dump_snapshot, mock_record_memory_history, mock_annotations
    ):
        with tempfile.TemporaryDirectory() as output_dir:
            manager = self._make_memory_manager(output_dir)

            output = manager._stop_profile()

        self.assertFalse(output.success)
        self.assertIn("stoi", output.message)
        mock_dump_snapshot.assert_called_once()
        mock_record_memory_history.assert_called_once_with(enabled=None)
        self.assertFalse(manager.profile_in_progress)
        self.assertIsNone(manager.profiler_start_forward_ct)
        mock_annotations.assert_called_once_with(False)

    @patch(
        "sglang.srt.managers.scheduler_components.profiler_manager.set_detailed_annotations_enabled"
    )
    @patch(
        "sglang.srt.managers.scheduler_components.profiler_manager.torch.cuda.memory._record_memory_history",
        side_effect=RuntimeError("cleanup failed"),
    )
    @patch(
        "sglang.srt.managers.scheduler_components.profiler_manager.torch.cuda.memory._dump_snapshot",
        side_effect=RuntimeError("stoi"),
    )
    def test_cleanup_failure_does_not_mask_snapshot_failure(
        self, mock_dump_snapshot, mock_record_memory_history, mock_annotations
    ):
        with tempfile.TemporaryDirectory() as output_dir:
            manager = self._make_memory_manager(output_dir)

            output = manager._stop_profile()

        self.assertFalse(output.success)
        self.assertIn("stoi", output.message)
        mock_dump_snapshot.assert_called_once()
        mock_record_memory_history.assert_called_once_with(enabled=None)
        self.assertFalse(manager.profile_in_progress)
        self.assertIsNone(manager.profiler_start_forward_ct)
        mock_annotations.assert_called_once_with(False)


if __name__ == "__main__":
    unittest.main()
