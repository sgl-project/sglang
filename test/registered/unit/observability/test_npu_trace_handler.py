# SPDX-License-Identifier: Apache-2.0
"""Shared NPU trace policy and both SRT callers, with real CPU profiling."""

import multiprocessing
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.scheduler_components import profiler_manager
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.observability import profile_analysis_npu, profile_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

TORCH_PROFILE = torch.profiler.profile


class TestNPUTraceHandler(unittest.TestCase):
    def test_dependency_import_requires_available_npu(self):
        with patch("sglang.srt.utils.is_npu", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires an available NPU"):
                profile_analysis_npu.get_torch_npu()

    def test_policy_depends_on_process_not_workload_name(self):
        api = SimpleNamespace(tensorboard_trace_handler=Mock())
        profiler = SimpleNamespace(prof_if=SimpleNamespace(prof_path="rank1_ascend_pt"))
        for daemon in (False, True):
            with (
                self.subTest(daemon=daemon),
                patch.object(
                    multiprocessing,
                    "current_process",
                    return_value=SimpleNamespace(daemon=daemon),
                ),
                patch.object(
                    profile_analysis_npu,
                    "get_torch_npu",
                    return_value=SimpleNamespace(profiler=api),
                ),
                patch.object(
                    profile_analysis_npu,
                    "start_npu_profile_analysis",
                    return_value=Path("sglang_analysis.json"),
                ) as analyze,
            ):
                handler = profile_analysis_npu.NPUTraceHandler("traces")
                self.assertEqual(handler.npu_auto_analysis_in_daemon, daemon)
                api.tensorboard_trace_handler.assert_called_with(
                    "traces", analyse_flag=not daemon
                )
                self.assertIs(
                    handler.on_trace_ready, api.tensorboard_trace_handler.return_value
                )
                trace = handler.export(profiler)
                if daemon:
                    analyze.assert_called_once_with("rank1_ascend_pt")
                    self.assertEqual(trace, Path("sglang_analysis.json"))
                else:
                    analyze.assert_not_called()
                    self.assertIsNone(trace)

    def test_background_launch_failure_is_logged_not_raised(self):
        profiler = SimpleNamespace(prof_if=SimpleNamespace(prof_path="rank0_ascend_pt"))
        handler = profile_analysis_npu.NPUTraceHandler.__new__(
            profile_analysis_npu.NPUTraceHandler
        )
        handler.npu_auto_analysis_in_daemon = True
        with (
            patch.object(
                profile_analysis_npu,
                "start_npu_profile_analysis",
                side_effect=OSError("cannot exec"),
            ),
            self.assertLogs(profile_analysis_npu.logger, level="ERROR") as logs,
        ):
            self.assertIsNone(handler.export(profiler))
        self.assertIn("cannot exec", "\n".join(logs.output))


class TestSRTNPUTraceHandler(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory(dir=Path.cwd()))
        self.events = []
        self.native_analysis = Mock(
            side_effect=lambda: self.events.append("native analysis")
        )

        def trace_handler(directory, analyse_flag=True):
            def trace_ready(profiler):
                profiler.prof_if.prof_path = str(Path(directory) / "rank0_ascend_pt")
                if analyse_flag:
                    self.native_analysis()

            return trace_ready

        api = SimpleNamespace(
            tensorboard_trace_handler=Mock(side_effect=trace_handler),
            _ExperimentalConfig=Mock(return_value=None),
            ExportType=SimpleNamespace(Text="text"),
            ProfilerLevel=SimpleNamespace(Level1=1),
            AiCMetrics=SimpleNamespace(PipeUtilization=1),
        )
        self.api = api
        npu = SimpleNamespace(profiler=api)
        self.get_npu = self.enterContext(
            patch.object(profile_analysis_npu, "get_torch_npu", return_value=npu)
        )
        self.enterContext(patch.object(profiler_manager, "torch_npu", npu, create=True))
        self.enterContext(patch.object(profiler_manager, "_is_npu", True))
        self.enterContext(patch.object(profile_utils, "_is_npu", True))
        self.enterContext(
            patch.object(
                profile_utils.current_platform, "is_out_of_tree", return_value=False
            )
        )
        self.enterContext(
            patch.object(
                profiler_manager,
                "get_dp_tp_group",
                return_value=SimpleNamespace(cpu_group=None),
            )
        )
        self.enterContext(
            patch.object(
                profile_utils,
                "get_device",
                return_value=SimpleNamespace(gpu_id=0, base_gpu_id=0),
            )
        )
        parallel = SimpleNamespace(tp_rank=0, num_dp_ranks=1, pp_size=1, moe_ep_size=1)
        self.enterContext(
            patch.object(profile_utils, "get_parallel", return_value=parallel)
        )
        self.enterContext(
            patch.object(profiler_manager, "get_parallel", return_value=parallel)
        )
        self.enterContext(
            patch.object(
                torch.distributed,
                "barrier",
                side_effect=lambda *_: self.events.append("barrier"),
            )
        )
        self.process = self.enterContext(
            patch.object(
                multiprocessing,
                "current_process",
                return_value=SimpleNamespace(daemon=False),
            )
        )
        self.analyze = self.enterContext(
            patch.object(
                profile_analysis_npu,
                "start_npu_profile_analysis",
                side_effect=lambda _: self.events.append("submit analysis"),
            )
        )

        def cpu_profile(**kwargs):
            profiler = TORCH_PROFILE(**kwargs)
            profiler.prof_if = SimpleNamespace()
            return profiler

        self.enterContext(
            patch.object(torch.profiler, "profile", side_effect=cpu_profile)
        )

    def run_profile(self, v2):
        with patch.object(
            profiler_manager.envs.SGLANG_PROFILE_V2, "get", return_value=v2
        ):
            manager = profiler_manager.SchedulerProfilerManager(
                get_forward_ct=lambda: 0
            )
            result = manager._init_profile(
                self.directory,
                None,
                1,
                ["CPU"],
                False,
                False,
                v2,
                "test",
                profile_stages=["prefill"],
            )
            self.assertTrue(result.success)
            if v2:
                manager._profile_manager.step(ForwardMode.EXTEND)
                torch.ones(1).add_(1)
                manager._profile_manager.step(ForwardMode.DECODE)
                self.assertIsNone(manager._profile_manager.profiler)
            else:
                self.assertTrue(manager._start_profile().success)
                torch.ones(1).add_(1)
                self.assertTrue(manager._stop_profile().success)
                self.assertFalse(manager.profile_in_progress)

    def test_both_srt_paths_keep_native_non_daemon_analysis(self):
        for v2 in (False, True):
            with self.subTest(v2=v2):
                self.events.clear()
                self.native_analysis.reset_mock()
                self.run_profile(v2)
                self.api.tensorboard_trace_handler.assert_called_with(
                    self.directory, analyse_flag=True
                )
                self.native_analysis.assert_called_once()
                self.analyze.assert_not_called()
                self.assertEqual(self.events, ["native analysis", "barrier"])

    def test_both_srt_paths_use_shared_daemon_export_after_stop(self):
        self.process.return_value.daemon = True
        for v2 in (False, True):
            with self.subTest(v2=v2):
                self.events.clear()
                self.analyze.reset_mock()
                self.run_profile(v2)
                self.api.tensorboard_trace_handler.assert_called_with(
                    self.directory, analyse_flag=False
                )
                self.analyze.assert_called_once_with(
                    str(Path(self.directory) / "rank0_ascend_pt")
                )
                self.native_analysis.assert_not_called()
                self.assertEqual(self.events, ["submit analysis", "barrier"])

    def test_non_npu_srt_paths_keep_chrome_trace_export(self):
        with (
            patch.object(profiler_manager, "_is_npu", False),
            patch.object(profile_utils, "_is_npu", False),
        ):
            for v2 in (False, True):
                with self.subTest(v2=v2):
                    self.run_profile(v2)
        self.get_npu.assert_not_called()
        self.analyze.assert_not_called()
        traces = list(Path(self.directory).glob("*.trace.json.gz"))
        self.assertEqual(len(traces), 2)
        self.assertTrue(all(path.stat().st_size > 0 for path in traces))


if __name__ == "__main__":
    unittest.main()
