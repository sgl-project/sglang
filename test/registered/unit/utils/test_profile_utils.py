import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.utils import profile_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _platform(*, xpu: bool, out_of_tree: bool = False):
    return SimpleNamespace(
        is_xpu=lambda: xpu,
        is_out_of_tree=lambda: out_of_tree,
        get_torch_profiler_activity_str=lambda: "XPU",
        get_torch_profiler_activity=lambda: torch.profiler.ProfilerActivity.XPU,
    )


class TestTorchProfilerActivityMap(CustomTestCase):
    def test_missing_out_of_tree_activity_is_ignored(self):
        platform = SimpleNamespace(
            is_xpu=lambda: False,
            is_out_of_tree=lambda: True,
            get_torch_profiler_activity_str=lambda: "PRIVATEUSE1",
            get_torch_profiler_activity=Mock(
                side_effect=AssertionError("unsupported activity")
            ),
        )

        with patch.object(profile_utils, "current_platform", platform):
            activity_map = profile_utils.get_torch_profiler_activity_map()

        self.assertNotIn("PRIVATEUSE1", activity_map)
        platform.get_torch_profiler_activity.assert_not_called()

    def test_explicit_xpu_key_survives_on_a_non_xpu_platform(self):
        # The generic "GPU" stays CUDA off XPU, but a caller naming "XPU"
        # explicitly must still resolve when the build exposes that activity.
        with patch.object(profile_utils, "current_platform", _platform(xpu=False)):
            activity_map = profile_utils.get_torch_profiler_activity_map()

        self.assertIs(activity_map["GPU"], torch.profiler.ProfilerActivity.CUDA)
        if hasattr(torch.profiler.ProfilerActivity, "XPU"):
            self.assertIs(activity_map["XPU"], torch.profiler.ProfilerActivity.XPU)


class TestResolveTorchProfilerActivities(CustomTestCase):
    def test_aliased_names_collapse_to_one_activity(self):
        # On XPU both "GPU" and "XPU" mean the XPU activity. torch.profiler
        # raises on a repeated activity by 2.13, so the duplicate has to be
        # gone before the list reaches the constructor.
        with patch.object(profile_utils, "current_platform", _platform(xpu=True)):
            resolved = profile_utils.resolve_torch_profiler_activities(
                ["CPU", "GPU", "XPU"]
            )

        self.assertEqual(
            resolved,
            [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ],
        )

    def test_request_order_is_preserved_and_unknown_names_dropped(self):
        with patch.object(profile_utils, "current_platform", _platform(xpu=True)):
            resolved = profile_utils.resolve_torch_profiler_activities(
                ["GPU", "MEM", "CPU"]
            )

        self.assertEqual(
            resolved,
            [
                torch.profiler.ProfilerActivity.XPU,
                torch.profiler.ProfilerActivity.CPU,
            ],
        )


class TestTorchProfilerRouting(CustomTestCase):
    def test_legacy_profiler_routes_generic_gpu_to_xpu(self):
        manager = object.__new__(SchedulerProfilerManager)
        manager.profiler_activities = ["GPU"]
        manager.torch_profiler_with_stack = False
        manager.torch_profiler_record_shapes = False
        manager.torch_profiler_output_dir = Path("/tmp")
        manager.profile_id = "xpu-test"
        manager.ps = SimpleNamespace(tp_rank=0, gpu_id=0)
        manager.profile_in_progress = False
        profiler = Mock()

        with (
            patch.object(profile_utils, "current_platform", _platform(xpu=True)),
            patch.object(torch.profiler, "profile", return_value=profiler) as create,
        ):
            result = manager._start_profile()

        self.assertTrue(result.success)
        self.assertEqual(
            create.call_args.kwargs["activities"],
            [torch.profiler.ProfilerActivity.XPU],
        )
        profiler.start.assert_called_once_with()

    def test_v2_profiler_routes_generic_gpu_to_xpu(self):
        profiler_impl = profile_utils._ProfilerTorch(
            output_dir="/tmp",
            output_prefix="",
            output_suffix="",
            profile_id="xpu-test",
            ps=SimpleNamespace(tp_rank=0),
            cpu_group=None,
            first_rank_in_node=True,
            activities=["GPU"],
            with_stack=False,
            record_shapes=False,
        )
        profiler = Mock()

        with (
            patch.object(profile_utils, "current_platform", _platform(xpu=True)),
            patch.object(torch.profiler, "profile", return_value=profiler) as create,
        ):
            profiler_impl.start()

        self.assertEqual(
            create.call_args.kwargs["activities"],
            [torch.profiler.ProfilerActivity.XPU],
        )
        profiler.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
