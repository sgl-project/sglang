# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for patched profiler construction and NPU callback ownership."""

import importlib.util
import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.utils import profiler as profiler_module
from sglang.srt.observability import profile_analysis_npu

TORCH_PROFILE = torch.profiler.profile
TORCH_SCHEDULE = torch.profiler.schedule


class FakeNPUProfile(TORCH_PROFILE):
    """Exercise the real torch schedule with the NPU callback's extra interface."""

    def __init__(self, **kwargs):
        kwargs["activities"] = [torch.profiler.ProfilerActivity.CPU]
        self.prof_if = SimpleNamespace(analyse=Mock())
        super().__init__(**kwargs)


def cpu_profile(**kwargs):
    # The fake NPU activity maps to CPU: deduplicate it on the old, buggy path.
    kwargs["activities"] = list(dict.fromkeys(kwargs["activities"]))
    return TORCH_PROFILE(**kwargs)


class TestDiffusionProfiler(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory(dir=Path.cwd()))
        self.npu_platform = self.enterContext(
            patch.object(profiler_module.current_platform, "is_npu", return_value=True)
        )
        self.enterContext(patch.object(torch.cuda, "is_available", return_value=False))
        if hasattr(torch, "xpu"):
            self.enterContext(
                patch.object(torch.xpu, "is_available", return_value=False)
            )
        if hasattr(torch, "musa"):
            self.enterContext(
                patch.object(torch.musa, "is_available", return_value=False)
            )
        self.enterContext(
            patch.object(torch, "npu", SimpleNamespace(synchronize=Mock()), create=True)
        )
        self.callbacks = []
        self.process = self.enterContext(
            patch.object(
                multiprocessing,
                "current_process",
                return_value=SimpleNamespace(daemon=False),
            )
        )

        def trace_handler(directory, analyse_flag=True):
            def trace_ready(instance):
                instance.prof_if.prof_path = str(Path(directory) / "rank0_ascend_pt")
                if analyse_flag:
                    if self.process.return_value.daemon:
                        raise RuntimeError(
                            "Cannot parse profiling data in a daemon process"
                        )
                    instance.prof_if.analyse()
                self.callbacks.append(instance)

            return trace_ready

        self.npu_api = SimpleNamespace(
            profile=Mock(side_effect=FakeNPUProfile),
            schedule=Mock(wraps=TORCH_SCHEDULE),
            ProfilerActivity=SimpleNamespace(
                CPU=torch.profiler.ProfilerActivity.CPU,
                NPU=torch.profiler.ProfilerActivity.CPU,
            ),
            tensorboard_trace_handler=Mock(side_effect=trace_handler),
        )
        self.torch_profile = self.enterContext(
            patch.object(torch.profiler, "profile", side_effect=cpu_profile)
        )
        self.torch_schedule = self.enterContext(
            patch.object(torch.profiler, "schedule", wraps=TORCH_SCHEDULE)
        )

        def apply_patches(patches):
            for destination, replacement in patches:
                self.assertIn(destination, ("profiler.profile", "profiler.schedule"))
                setattr(torch.profiler, destination.split(".")[-1], replacement)

        self.apply_patches = Mock(side_effect=apply_patches)
        self.torch_npu = SimpleNamespace(
            profiler=self.npu_api,
            _apply_all_patches=Mock(),
            _init=SimpleNamespace(
                patches=SimpleNamespace(
                    monkey_patches=SimpleNamespace(_apply_patches=self.apply_patches)
                )
            ),
        )
        self.enterContext(patch.dict(sys.modules, torch_npu=self.torch_npu))
        self.enterContext(patch("sglang.srt.utils.is_npu", return_value=True))
        self.profiler_utils = self.load_profiler_module(npu=True)
        self.analyze = self.enterContext(
            patch.object(profile_analysis_npu, "analyze_npu_profile")
        )
        self.parallel_initialized = self.enterContext(
            patch.object(
                self.profiler_utils,
                "model_parallel_is_initialized",
                return_value=False,
            )
        )

    def load_profiler_module(self, npu):
        # Execute the real module's import-time patch call in an isolated namespace.
        # Restore torch attributes after each test so collection order cannot leak patches.
        spec = importlib.util.spec_from_file_location(
            "diffusion_profiler_under_test", profiler_module.__file__
        )
        module = importlib.util.module_from_spec(spec)
        with patch.object(profiler_module.current_platform, "is_npu", return_value=npu):
            spec.loader.exec_module(module)
        return module

    def make_profiler(self, full=False, steps=5, rank=0):
        wrapper = self.profiler_utils.SGLDiffusionProfiler(
            full_profile=full,
            num_steps=steps,
            num_inference_steps=40,
            log_dir=self.directory,
            rank=rank,
        )
        self.addCleanup(wrapper.stop, export_trace=False)
        return wrapper

    def test_npu_scheduled_callback_at_seventh_step(self):
        wrapper = self.make_profiler()
        for step in range(7):
            torch.ones(1).add_(1)
            wrapper.step_denoising_step()
            self.assertEqual(len(self.callbacks), int(step == 6))
        wrapper.step_denoising_step()
        wrapper.stop()  # Idempotent after the short window stops itself.
        self.assertTrue(wrapper.has_stopped)
        self.assertIsNone(self.profiler_utils.SGLDiffusionProfiler.get_instance())
        self.assertIs(self.callbacks[0], wrapper.profiler)
        self.apply_patches.assert_called_once()
        self.torch_npu._apply_all_patches.assert_not_called()
        self.npu_api.profile.assert_called_once()
        self.npu_api.schedule.assert_called_once()
        self.torch_profile.assert_not_called()
        self.torch_schedule.assert_not_called()

    def test_npu_full_profile_records_both_steps_without_schedule_warmup(self):
        wrapper = self.make_profiler(full=True)
        for step in range(2):
            with torch.profiler.record_function(f"denoising_step_{step}"):
                torch.ones(1).add_(1)
        wrapper.step_stage()
        wrapper.stop()
        names = {event.name for event in wrapper.profiler.events()}
        self.assertTrue({"denoising_step_0", "denoising_step_1"} <= names)
        self.assertEqual(self.callbacks, [wrapper.profiler])
        self.npu_api.profile.assert_called_once()
        self.npu_api.schedule.assert_not_called()
        self.torch_profile.assert_not_called()

    def test_npu_whole_denoising_window_exports_on_stop(self):
        wrapper = self.make_profiler(steps=-1)
        for _ in range(40):
            torch.ones(1).add_(1)
            wrapper.step_denoising_step()
        wrapper.stop()
        self.assertEqual(self.callbacks, [wrapper.profiler])
        self.npu_api.schedule.assert_called_once_with(
            skip_first=0, wait=0, warmup=1, active=41, repeat=1
        )

    def test_daemon_worker_automatically_analyzes_on_stop(self):
        self.process.return_value.daemon = True
        for full in (False, True):
            with self.subTest(full=full):
                self.analyze.reset_mock()
                wrapper = self.make_profiler(full=full)
                if not full:
                    wrapper.step_denoising_step()
                torch.ones(1).add_(1)
                wrapper.stop()
                self.analyze.assert_called_once_with(wrapper.profiler.prof_if.prof_path)
                wrapper.stop()  # Request cleanup must not parse a second time.
                self.analyze.assert_called_once()
                self.npu_api.tensorboard_trace_handler.assert_called_with(
                    self.directory, analyse_flag=False
                )
                wrapper.profiler.prof_if.analyse.assert_not_called()

    def test_short_capture_is_parsed_before_generation_continues(self):
        self.process.return_value.daemon = True
        wrapper = self.make_profiler()
        events = []

        def analyze(path):
            self.assertTrue(wrapper.has_stopped)
            self.assertEqual(path, wrapper.profiler.prof_if.prof_path)
            events.append("analysis")

        self.analyze.side_effect = analyze
        for step in range(8):
            torch.ones(1).add_(1)
            wrapper.step_denoising_step()
            if step < 7:
                self.analyze.assert_not_called()
        events.append("remaining generation")
        self.assertEqual(events, ["analysis", "remaining generation"])
        self.assertIsNone(self.profiler_utils.SGLDiffusionProfiler.get_instance())
        wrapper.stop()
        self.analyze.assert_called_once()

    def test_daemon_non_driver_exports_despite_dump_rank_zero(self):
        self.process.return_value.daemon = True
        wrapper = self.make_profiler(full=True, rank=1)
        wrapper.stop(dump_rank=0)
        self.analyze.assert_called_once_with(wrapper.profiler.prof_if.prof_path)

    def test_error_cleanup_skips_daemon_analysis_and_collectives(self):
        self.process.return_value.daemon = True
        self.parallel_initialized.return_value = True
        wrapper = self.make_profiler(full=True)
        with patch.object(torch.distributed, "all_gather_object") as gather:
            wrapper.stop(export_trace=False)
        gather.assert_not_called()
        self.analyze.assert_not_called()
        self.assertTrue(wrapper.has_stopped)
        self.assertIsNone(self.profiler_utils.SGLDiffusionProfiler.get_instance())

    def test_analysis_failure_is_not_hidden_by_native_callback(self):
        self.process.return_value.daemon = True
        self.analyze.side_effect = RuntimeError("No parsed trace")
        wrapper = self.make_profiler(full=True)
        torch.ones(1).add_(1)
        with self.assertRaisesRegex(RuntimeError, "No parsed trace"):
            wrapper.stop()
        self.assertIsNone(self.profiler_utils.SGLDiffusionProfiler.get_instance())

    def test_stop_failure_uses_the_same_analysis_error_path(self):
        self.process.return_value.daemon = True
        self.parallel_initialized.return_value = True
        group = SimpleNamespace(world_size=2, cpu_group=object())
        wrapper = self.make_profiler(full=True)
        try:
            with (
                patch.object(
                    wrapper.profiler, "stop", side_effect=RuntimeError("stop failed")
                ),
                patch.object(
                    self.profiler_utils, "get_replica_group", return_value=group
                ),
                patch.object(torch.distributed, "all_gather_object") as gather,
            ):
                gather.side_effect = lambda errors, local, group: errors.__setitem__(
                    slice(None), [local, None]
                )
                with self.assertRaisesRegex(RuntimeError, "rank 0: stop failed"):
                    wrapper.stop()
                gather.assert_called_once()
            self.analyze.assert_not_called()
        finally:
            wrapper.profiler.stop()  # Release the real CPU profiler after injected failure.

    def test_analysis_failure_on_another_rank_is_reported(self):
        self.process.return_value.daemon = True
        self.parallel_initialized.return_value = True
        group = SimpleNamespace(world_size=2, cpu_group=object())
        with (
            patch.object(self.profiler_utils, "get_replica_group", return_value=group),
            patch.object(torch.distributed, "all_gather_object") as gather,
        ):
            gather.side_effect = lambda errors, local, group: errors.__setitem__(
                slice(None), [local, "rank 1 parser failed"]
            )
            wrapper = self.make_profiler(full=True)
            torch.ones(1).add_(1)
            with self.assertRaisesRegex(RuntimeError, "rank 1 parser failed"):
                wrapper.stop()
            self.assertIs(gather.call_args.kwargs["group"], group.cpu_group)

    def test_non_npu_keeps_torch_profiler_and_chrome_export(self):
        self.npu_platform.return_value = False
        torch.profiler.profile = self.torch_profile
        torch.profiler.schedule = self.torch_schedule
        self.apply_patches.reset_mock()
        self.profiler_utils = self.load_profiler_module(npu=False)
        for full in (False, True):
            with self.subTest(full=full):
                wrapper = self.make_profiler(full=full)
                if not full:
                    wrapper.step_denoising_step()
                torch.ones(1).add_(1)
                wrapper.stop()
                wrapper.stop()
                self.assertNotIsInstance(wrapper.profiler, FakeNPUProfile)
        self.apply_patches.assert_not_called()
        self.assertEqual(self.torch_profile.call_count, 2)
        self.torch_schedule.assert_called_once()
        self.npu_api.profile.assert_not_called()
        self.npu_api.schedule.assert_not_called()
        self.npu_api.tensorboard_trace_handler.assert_not_called()
        traces = list(Path(self.directory).glob("*.trace.json.gz"))
        self.assertEqual(len(traces), 2)
        self.assertTrue(all(path.stat().st_size > 0 for path in traces))


if __name__ == "__main__":
    unittest.main()
