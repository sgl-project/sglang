"""Unit tests for MLX Metal profiling patch (hardware_backend/mlx/profiler.py).

Covers:
  - apply_metal_profiler_patches() replaces torch.profiler.profile
  - MLX path: MetalTorchProfiler.start/stop produces a .gputrace file
  - MPS path: MetalTorchProfiler wraps torch.mps.profiler.metal_capture
  - RuntimeError from start_capture is caught and returned as success=False
  - SchedulerProfilerManager._start_profile returns success=False gracefully
    when Metal capture fails (no MTL_CAPTURE_ENABLED)

Skips on non-Apple-Silicon platforms and when ``mlx`` is missing.
"""

from __future__ import annotations

import importlib.util
import platform
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires Apple Silicon and mlx"


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestApplyMetalProfilerPatches(unittest.TestCase):
    """apply_metal_profiler_patches() replaces torch.profiler.profile."""

    def setUp(self):
        import torch

        self._original_profile = getattr(
            torch.profiler.profile, "_sglang_original_profile", None
        )

    def tearDown(self):
        import torch

        if self._original_profile is not None:
            torch.profiler.profile = self._original_profile

    def test_patch_replaces_profile(self):
        import torch

        from sglang.srt.hardware_backend.mlx.profiler import (
            MetalTorchProfiler,
            apply_metal_profiler_patches,
        )

        apply_metal_profiler_patches()
        self.assertTrue(getattr(torch.profiler.profile, "_sglang_metal_patched", False))
        p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
        self.assertIsInstance(p, MetalTorchProfiler)

    def test_patch_is_idempotent(self):
        import torch

        from sglang.srt.hardware_backend.mlx.profiler import (
            apply_metal_profiler_patches,
        )

        apply_metal_profiler_patches()
        first = torch.profiler.profile
        apply_metal_profiler_patches()
        self.assertIs(torch.profiler.profile, first)

    def test_no_cuda_activity_uses_original(self):
        import torch

        from sglang.srt.hardware_backend.mlx.profiler import (
            MetalTorchProfiler,
            apply_metal_profiler_patches,
        )

        apply_metal_profiler_patches()
        p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        self.assertNotIsInstance(p, MetalTorchProfiler)


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMetalCaptureProfilerMLX(unittest.TestCase):
    """MLX path: start_mlx produces a .gputrace and stop_capture is called."""

    def test_start_mlx_success(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.profiler import MetalCaptureProfiler

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "test.gputrace"
            with patch.object(mx.metal, "start_capture"), patch.object(
                mx.metal, "stop_capture"
            ):
                profiler, result = MetalCaptureProfiler.start_mlx(trace_path)

            self.assertTrue(result.success)
            self.assertIsNotNone(profiler)
            self.assertEqual(profiler.label, "MLX")
            self.assertTrue(profiler.standalone)

    def test_start_mlx_runtime_error_returns_failure(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.profiler import MetalCaptureProfiler

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "test.gputrace"
            with patch.object(
                mx.metal,
                "start_capture",
                side_effect=RuntimeError("Capture layer is not inserted"),
            ):
                profiler, result = MetalCaptureProfiler.start_mlx(trace_path)

        self.assertIsNone(profiler)
        self.assertFalse(result.success)
        self.assertIn("MTL_CAPTURE_ENABLED", result.message)

    def test_stop_calls_stop_capture(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.profiler import MetalCaptureProfiler

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "test.gputrace"
            with patch.object(mx.metal, "start_capture"), patch.object(
                mx.metal, "stop_capture"
            ) as mock_stop:
                profiler, _ = MetalCaptureProfiler.start_mlx(trace_path)
                profiler.stop()
                mock_stop.assert_called_once()


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMetalCaptureProfilerMPS(unittest.TestCase):
    """MPS path: start_mps wraps torch.mps.profiler.metal_capture."""

    def test_start_mps_success(self):
        import torch

        from sglang.srt.hardware_backend.mlx.profiler import MetalCaptureProfiler

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "test.gputrace"
            with patch.object(
                torch.mps.profiler, "metal_capture", return_value=mock_ctx
            ):
                profiler, result = MetalCaptureProfiler.start_mps(trace_path)

        self.assertTrue(result.success)
        self.assertIsNotNone(profiler)
        self.assertEqual(profiler.label, "MPS")
        self.assertFalse(profiler.standalone)

    def test_start_mps_runtime_error_returns_failure(self):
        import torch

        from sglang.srt.hardware_backend.mlx.profiler import MetalCaptureProfiler

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "test.gputrace"
            with patch.object(
                torch.mps.profiler,
                "metal_capture",
                side_effect=RuntimeError("MPS profiler unavailable"),
            ):
                profiler, result = MetalCaptureProfiler.start_mps(trace_path)

        self.assertIsNone(profiler)
        self.assertFalse(result.success)
        self.assertIn("MTL_CAPTURE_ENABLED", result.message)


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestSchedulerProfilerManagerMPS(unittest.TestCase):
    """SchedulerProfilerManager._start_profile handles Metal capture failures."""

    def _make_manager(self, output_dir):
        from sglang.srt.managers.scheduler_components.profiler_manager import (
            SchedulerProfilerManager,
        )

        class FakePS:
            tp_rank = dp_rank = pp_rank = moe_ep_rank = 0
            dp_size = pp_size = moe_ep_size = 1
            gpu_id = 0

        mgr = SchedulerProfilerManager(
            ps=FakePS(), dp_tp_cpu_group=None, get_forward_ct=lambda: 0
        )
        mgr._init_profile(output_dir, None, None, None, None, None, False, "test")
        return mgr

    def test_start_profile_failure_does_not_crash(self):
        import torch

        from sglang.srt.hardware_backend.mlx.profiler import (
            apply_metal_profiler_patches,
        )

        apply_metal_profiler_patches()

        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            with (
                patch(
                    "sglang.srt.hardware_backend.mlx.profiler.use_mlx",
                    return_value=False,
                ),
                patch.object(
                    torch.mps.profiler,
                    "metal_capture",
                    side_effect=RuntimeError("Capture layer is not inserted"),
                ),
            ):
                result = mgr._start_profile()

        self.assertFalse(result.success)
        self.assertFalse(mgr.profile_in_progress)
        self.assertIsNone(mgr.torch_profiler)

    def test_start_profile_success_with_mock_capture(self):
        import torch

        from sglang.srt.hardware_backend.mlx.profiler import (
            apply_metal_profiler_patches,
        )

        apply_metal_profiler_patches()

        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            mock_ctx = MagicMock()
            with (
                patch(
                    "sglang.srt.hardware_backend.mlx.profiler.use_mlx",
                    return_value=False,
                ),
                patch.object(
                    torch.mps.profiler, "metal_capture", return_value=mock_ctx
                ),
                patch("torch.distributed.barrier"),
            ):
                result = mgr._start_profile()
                self.assertTrue(result.success)
                self.assertTrue(mgr.profile_in_progress)
                mgr._stop_profile()
                self.assertFalse(mgr.profile_in_progress)

            mock_ctx.__enter__.assert_called_once_with()
            mock_ctx.__exit__.assert_called_once_with(None, None, None)


if __name__ == "__main__":
    unittest.main()
