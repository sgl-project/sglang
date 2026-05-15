import tempfile
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.compilation.export_artifact import (
    make_export_artifact_spec,
)
from sglang.srt.compilation.export_backends import ExportRuntime
from sglang.srt.compilation.torch_compile import TorchCompileConfig, sgl_compile
from sglang.srt.environ import envs
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class _StubRuntime(ExportRuntime):
    def __init__(self, runtime_callable):
        self.runtime_callable = runtime_callable
        self.calls = []

    def prepare_runtime(self, exported_program, artifact, compile_config):
        self.calls.append((exported_program, artifact, compile_config))
        return self.runtime_callable


class _StubPlatform(SRTPlatform):
    _enum = PlatformEnum.CUDA
    device_name = "cuda"
    device_type = "cuda"

    def __init__(
        self,
        strategy="compile",
        defaults: TorchCompileConfig | None = None,
        runtime_callable=None,
    ):
        self.strategy = strategy
        self.defaults = defaults or TorchCompileConfig()
        self.runtime_callable = runtime_callable
        self.runtime = _StubRuntime(runtime_callable) if runtime_callable else None

    def get_compile_backend(self, mode=None):
        return f"backend:{mode}" if mode else "backend"

    def torch_compile_strategy(self):
        return self.strategy

    def torch_compile_defaults(self):
        return self.defaults

    def get_export_runtime(self, compile_config):
        if self.runtime is not None:
            return self.runtime
        return super().get_export_runtime(compile_config)

    def get_device_total_memory(self, device_id=0):
        return 0

    def get_current_memory_usage(self, device=None):
        return 0.0


class TestSglCompile(CustomTestCase):
    def setUp(self):
        import sglang.srt.platforms as platforms

        self.platforms = platforms
        self.saved_platform = platforms._current_platform
        platforms._current_platform = _StubPlatform()

    def tearDown(self):
        self.platforms._current_platform = self.saved_platform

    def test_lazy_compile_uses_platform_backend(self):
        with patch("torch.compile", side_effect=lambda fn, **_: fn) as mock_compile:

            @sgl_compile(dynamic=True)
            def fn(x):
                return x + 1

            mock_compile.assert_not_called()
            self.assertEqual(fn(1), 2)

        mock_compile.assert_called_once()
        self.assertEqual(mock_compile.call_args.kwargs["backend"], "backend")
        self.assertTrue(mock_compile.call_args.kwargs["dynamic"])
        self.assertFalse(mock_compile.call_args.kwargs["fullgraph"])

    def test_platform_forced_fields_override_decorator(self):
        self.platforms._current_platform = _StubPlatform(
            defaults=TorchCompileConfig(dynamic=False, forced_fields={"dynamic"})
        )

        with patch("torch.compile", side_effect=lambda fn, **_: fn) as mock_compile:

            @sgl_compile(dynamic=True, fullgraph=True)
            def fn(x):
                return x + 1

            self.assertEqual(fn(1), 2)

        self.assertFalse(mock_compile.call_args.kwargs["dynamic"])
        self.assertTrue(mock_compile.call_args.kwargs["fullgraph"])

    def test_disable_env_uses_original_function(self):
        with envs.SGLANG_DISABLE_TORCH_COMPILE.override(True):
            with patch("torch.compile") as mock_compile:

                @sgl_compile(dynamic=True)
                def fn(x):
                    return x + 1

                self.assertEqual(fn(1), 2)

        mock_compile.assert_not_called()

    def test_transient_compile_context_does_not_cache_noop(self):
        with patch("torch.compile", side_effect=lambda fn, **_: fn) as mock_compile:
            with patch(
                "sglang.srt.compilation.torch_compile.is_in_piecewise_cuda_graph",
                side_effect=[True, False, False],
            ):

                @sgl_compile(dynamic=True)
                def fn(x):
                    return x + 1

                self.assertEqual(fn(1), 2)
                mock_compile.assert_not_called()
                self.assertEqual(fn(2), 3)

        mock_compile.assert_called_once()

    def test_staticmethod_descriptor(self):
        with patch("torch.compile", side_effect=lambda fn, **_: fn) as mock_compile:

            class C:
                @sgl_compile(dynamic=True)
                @staticmethod
                def fn(x):
                    return x + 1

            self.assertEqual(C.fn(1), 2)

        mock_compile.assert_called_once()

    def test_export_capture_saves_and_runs_original_by_default(self):
        self.platforms._current_platform = _StubPlatform(strategy="export")
        exported_program = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with envs.SGLANG_EXPORT_DIR.override(tmpdir):
                with patch(
                    "torch.export.export", return_value=exported_program
                ) as mock_export:
                    with patch("torch.export.save") as mock_save:

                        @sgl_compile(key="unit_export")
                        def fn(x):
                            return x + 1

                        self.assertEqual(fn(1), 2)

        mock_export.assert_called_once()
        mock_save.assert_called_once()
        self.assertTrue(str(mock_save.call_args.args[1]).endswith("unit_export.pt2"))

    def test_export_run_exported_uses_platform_runtime(self):
        runtime = MagicMock(return_value="runtime")
        platform = _StubPlatform(
            strategy="export",
            runtime_callable=runtime,
        )
        self.platforms._current_platform = platform
        exported_program = MagicMock()

        with patch("torch.export.export", return_value=exported_program):

            @sgl_compile(run_exported=True)
            def fn(x):
                return x + 1

            self.assertEqual(fn(1), "runtime")

        runtime.assert_called_once_with(1)
        self.assertEqual(platform.runtime.calls[0][0], exported_program)
        self.assertEqual(platform.runtime.calls[0][1].export_format, "torch")

    def test_export_only_builds_artifacts_and_runs_original(self):
        self.platforms._current_platform = _StubPlatform(strategy="export")
        exported_program = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with envs.SGLANG_EXPORT_DIR.override(tmpdir):
                with patch("torch.export.export", return_value=exported_program):
                    with patch("torch.export.save") as mock_save:

                        @sgl_compile(key="unit_export_only", export_artifact_mode="export_only")
                        def fn(x):
                            return x + 1

                        self.assertEqual(fn(1), 2)

        mock_save.assert_called_once()

    def test_load_only_requires_prebuilt_torch_program(self):
        self.platforms._current_platform = _StubPlatform(strategy="export")

        with tempfile.TemporaryDirectory() as tmpdir:
            with envs.SGLANG_EXPORT_DIR.override(tmpdir):

                @sgl_compile(key="missing_export", export_artifact_mode="load_only")
                def fn(x):
                    return x + 1

                with self.assertRaises(FileNotFoundError):
                    fn(1)

    def test_artifact_spec_records_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with envs.SGLANG_EXPORT_DIR.override(tmpdir):
                spec = make_export_artifact_spec(
                    key="pkg.fn",
                    export_format="onnx",
                    mode="build_if_missing",
                    shape_policy="infer_dynamic",
                    copy_output_to_arg_index=0,
                    args=(torch.ones(2, 3),),
                )
                spec.write_metadata()
                loaded = spec.load_metadata()

        self.assertEqual(spec.safe_key, "pkg.fn")
        self.assertEqual(loaded.format, "onnx")
        self.assertEqual(loaded.copy_output_to_arg_index, 0)
        self.assertEqual(loaded.input_schema[0].shape, (2, 3))
        self.assertEqual(loaded.distributed_context["tp_size"], 1)

    def test_export_runtime_requirement_falls_back_to_compile(self):
        runtime = MagicMock(return_value="runtime")
        self.platforms._current_platform = _StubPlatform(
            strategy="export",
            runtime_callable=runtime,
        )

        with patch("torch.compile", side_effect=lambda fn, **_: fn) as mock_compile:
            with patch("torch.export.export") as mock_export:

                @sgl_compile(run_exported=True, requires_cuda_graph_safe=True)
                def fn(x):
                    return x + 1

                self.assertEqual(fn(1), 2)

        mock_compile.assert_called_once()
        mock_export.assert_not_called()
        runtime.assert_not_called()

    def test_export_runtime_requirement_can_error(self):
        self.platforms._current_platform = _StubPlatform(
            strategy="export",
            runtime_callable=MagicMock(),
        )

        @sgl_compile(
            run_exported=True,
            requires_cuda_graph_safe=True,
            export_fallback_strategy="error",
        )
        def fn(x):
            return x + 1

        with self.assertRaisesRegex(RuntimeError, "not CUDA graph capture safe"):
            fn(1)


if __name__ == "__main__":
    import unittest

    unittest.main()
