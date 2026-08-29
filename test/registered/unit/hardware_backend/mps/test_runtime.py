"""Tests for the single Torch-owned MPS runtime gate."""

import importlib.util
import subprocess
import sys
import types
import unittest
from importlib.metadata import PackageNotFoundError, version
from unittest import mock

import torch
from packaging.version import Version

from sglang.srt.hardware_backend.mps import runtime
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


def _fake_mlx(version: str | None, *, metal_available: bool = True):
    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    if version is not None:
        fake_core.__version__ = version
    fake_core.metal = types.SimpleNamespace(
        is_available=lambda: metal_available,
    )
    fake_mlx.core = fake_core
    return fake_mlx, fake_core


def _has_stable_distribution(distribution: str, series: tuple[int, int]) -> bool:
    try:
        installed = Version(version(distribution))
    except (PackageNotFoundError, ValueError):
        return False
    return not installed.is_prerelease and installed.release[:2] == series


class TestMpsRuntime(unittest.TestCase):
    def tearDown(self):
        runtime.validate_mps_runtime.cache_clear()

    def test_non_mps_server_does_not_import_mlx(self):
        script = """
import sys
from sglang.srt.server_args import ServerArgs
ServerArgs(model_path="dummy", device="cpu")
assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )

    def test_version_helpers(self):
        self.assertTrue(runtime._is_stable_series("0.32.7", (0, 32)))
        self.assertFalse(runtime._is_stable_series("0.31.9", (0, 32)))
        self.assertFalse(runtime._is_stable_series("0.33.0", (0, 32)))
        self.assertFalse(runtime._is_stable_series("0.32.1rc1", (0, 32)))
        self.assertFalse(runtime._is_stable_series("0.32.1.dev1", (0, 32)))
        self.assertFalse(runtime._is_stable_series("unknown", (0, 32)))

    def test_unvalidated_runtime_pairs_are_rejected(self):
        cases = (
            ("2.12.1", "0.32.0", "tested stable Torch 2.13.x"),
            ("2.13.0", "0.31.0", "MLX 0.32.x"),
            ("2.14.0", "0.32.0", "tested stable Torch 2.13.x"),
            ("2.13.0", "0.33.0", "tested stable Torch 2.13.x"),
            ("2.13.1rc1", "0.32.0", "tested stable Torch 2.13.x"),
            ("2.13.0", "0.32.1.dev1", "tested stable Torch 2.13.x"),
        )
        for torch_version, mlx_version, message in cases:
            with self.subTest(torch=torch_version, mlx=mlx_version):
                fake_mlx, fake_core = _fake_mlx(mlx_version)
                runtime.validate_mps_runtime.cache_clear()
                with (
                    mock.patch.dict(
                        sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}
                    ),
                    mock.patch.object(torch, "__version__", torch_version),
                    mock.patch.object(
                        torch.backends.mps, "is_available", return_value=True
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    runtime.validate_mps_runtime()

    def test_missing_mlx_version_has_an_actionable_error(self):
        fake_mlx, fake_core = _fake_mlx(None)
        with (
            mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}),
            mock.patch.object(torch, "__version__", "2.13.0"),
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
            self.assertRaisesRegex(RuntimeError, "MLX unknown"),
        ):
            runtime.validate_mps_runtime()

    def test_missing_mlx_has_an_actionable_error(self):
        with (
            mock.patch.dict(sys.modules, {"mlx": None, "mlx.core": None}),
            self.assertRaisesRegex(RuntimeError, "MLX is not installed"),
        ):
            runtime.validate_mps_runtime()

    def test_unavailable_metal_devices_have_actionable_errors(self):
        cases = (
            (False, True, "PyTorch MPS device"),
            (True, False, "MLX Metal device"),
        )
        for torch_mps_available, mlx_metal_available, message in cases:
            with self.subTest(message=message):
                fake_mlx, fake_core = _fake_mlx(
                    "0.32.0", metal_available=mlx_metal_available
                )
                runtime.validate_mps_runtime.cache_clear()
                with (
                    mock.patch.dict(
                        sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}
                    ),
                    mock.patch.object(torch, "__version__", "2.13.0"),
                    mock.patch.object(
                        torch.backends.mps,
                        "is_available",
                        return_value=torch_mps_available,
                    ),
                    mock.patch.object(
                        torch.mps, "compile_shader", mock.Mock(), create=True
                    ),
                    mock.patch.object(
                        torch.mps, "load_metallib", mock.Mock(), create=True
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    runtime.validate_mps_runtime()

    def test_missing_torch_metal_shader_compiler_has_an_actionable_error(self):
        fake_mlx, fake_core = _fake_mlx("0.32.0")
        with (
            mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}),
            mock.patch.object(torch, "__version__", "2.13.0"),
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
            mock.patch.object(torch.mps, "compile_shader", None, create=True),
            self.assertRaisesRegex(RuntimeError, "torch.mps.compile_shader"),
        ):
            runtime.validate_mps_runtime()

    def test_missing_torch_metallib_loader_has_an_actionable_error(self):
        fake_mlx, fake_core = _fake_mlx("0.32.0")
        with (
            mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}),
            mock.patch.object(torch, "__version__", "2.13.0"),
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
            mock.patch.object(torch.mps, "compile_shader", mock.Mock(), create=True),
            mock.patch.object(torch.mps, "load_metallib", None, create=True),
            self.assertRaisesRegex(RuntimeError, "torch.mps.load_metallib"),
        ):
            runtime.validate_mps_runtime()

    def test_missing_torch_mps_memory_api_has_an_actionable_error(self):
        for memory_api in ("recommended_max_memory", "driver_allocated_memory"):
            with self.subTest(memory_api=memory_api):
                fake_mlx, fake_core = _fake_mlx("0.32.0")
                runtime.validate_mps_runtime.cache_clear()
                with (
                    mock.patch.dict(
                        sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}
                    ),
                    mock.patch.object(torch, "__version__", "2.13.0"),
                    mock.patch.object(
                        torch.backends.mps, "is_available", return_value=True
                    ),
                    mock.patch.object(
                        torch.mps, "compile_shader", mock.Mock(), create=True
                    ),
                    mock.patch.object(
                        torch.mps, "load_metallib", mock.Mock(), create=True
                    ),
                    mock.patch.object(torch.mps, memory_api, None, create=True),
                    self.assertRaisesRegex(RuntimeError, memory_api),
                ):
                    runtime.validate_mps_runtime()

    def test_validated_runtime_accepts_patch_releases(self):
        fake_mlx, fake_core = _fake_mlx("0.32.9")
        with (
            mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}),
            mock.patch.object(torch, "__version__", "2.13.7"),
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
            mock.patch.object(torch.mps, "compile_shader", mock.Mock(), create=True),
            mock.patch.object(torch.mps, "load_metallib", mock.Mock(), create=True),
        ):
            self.assertIsNone(runtime.validate_mps_runtime())

    @unittest.skipUnless(
        importlib.util.find_spec("mlx") is not None
        and torch.backends.mps.is_available()
        and runtime._is_stable_series(torch.__version__, (2, 13))
        and _has_stable_distribution("mlx", (0, 32)),
        "requires the supported MPS runtime",
    )
    def test_current_runtime_is_supported(self):
        runtime.validate_mps_runtime()


if __name__ == "__main__":
    unittest.main()
