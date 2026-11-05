# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for patched profiler construction and NPU callback ownership."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.utils import profiler as profiler_module

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

        def trace_handler(directory, analyse_flag=True):
            def trace_ready(instance):
                instance.prof_if.prof_path = str(Path(directory) / "rank0_ascend_pt")
                if analyse_flag:
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
        self.profiler_utils = self.load_profiler_module(npu=True)

    def load_profiler_module(self, npu):
        # Execute the real module's import-time patch call in an isolated namespace.
        # Restore torch attributes after each test so collection order cannot leak patches.
        spec = importlib.util.spec_from_file_location(
            "diffusion_profiler_under_test", profiler_module.__file__
        )
        module = importlib.util.module_from_spec(spec)
        with (
            patch.dict(sys.modules, torch_npu=self.torch_npu),
            patch("sglang.srt.utils.is_npu", return_value=npu),
        ):
            spec.loader.exec_module(module)
        return module

    def make_profiler(self, full=False, steps=5):
        wrapper = self.profiler_utils.SGLDiffusionProfiler(
            full_profile=full,
            num_steps=steps,
            num_inference_steps=40,
            log_dir=self.directory,
        )
        self.addCleanup(wrapper.stop)
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
