"""Tests for the opt-in MLX runtime gate."""

import importlib.util
import os
import subprocess
import sys
import types
import unittest
from importlib.metadata import PackageNotFoundError, version
from unittest import mock

import torch
from packaging.version import Version

from sglang.srt.hardware_backend.mlx import runtime
from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


def _fake_mlx(version: str | None, *, metal_available: bool = True):
    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    if version is not None:
        fake_core.__version__ = version
    fake_core.metal = types.SimpleNamespace(is_available=lambda: metal_available)
    fake_mlx.core = fake_core
    return fake_mlx, fake_core


def _has_supported_mlx() -> bool:
    try:
        installed = Version(version("mlx"))
    except (PackageNotFoundError, ValueError):
        return False
    return not installed.is_prerelease and installed >= Version("0.32.0")


class TestMlxRuntime(unittest.TestCase):
    def test_disabled_backend_does_not_import_mlx(self):
        script = """
import sys
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.server_args import ServerArgs
assert use_mlx() is False
ServerArgs(model_path="dummy")
assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
"""
        env = os.environ.copy()
        env.pop("SGLANG_USE_MLX", None)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )

    def test_version_gates(self):
        self.assertTrue(runtime._is_stable_series("2.13.7", (2, 13)))
        self.assertFalse(runtime._is_stable_series("2.14.0", (2, 13)))
        self.assertFalse(runtime._is_stable_series("2.13.1rc1", (2, 13)))

        minimum = Version("0.32.0")
        self.assertTrue(runtime._is_stable_at_least("0.32.0", minimum))
        self.assertTrue(runtime._is_stable_at_least("0.32.0+local", minimum))
        self.assertTrue(runtime._is_stable_at_least("0.32.1.post1", minimum))
        self.assertTrue(runtime._is_stable_at_least("0.33.0", minimum))
        self.assertFalse(runtime._is_stable_at_least("0.31.9", minimum))
        self.assertFalse(runtime._is_stable_at_least("0.33.0rc1", minimum))
        self.assertFalse(runtime._is_stable_at_least("0.33.0.dev1", minimum))
        self.assertFalse(runtime._is_stable_at_least("unknown", minimum))

    def test_unvalidated_runtime_pairs_are_rejected(self):
        cases = (
            ("2.14.0", "0.32.0", "stable Torch 2.13.x"),
            ("2.13.0", "0.31.9", "MLX >= 0.32.0"),
            ("2.13.1rc1", "0.32.0", "stable Torch 2.13.x"),
            ("2.13.0", "0.33.0rc1", "MLX >= 0.32.0"),
            ("2.13.0", None, "MLX unknown"),
        )
        for torch_version, mlx_version, message in cases:
            with self.subTest(torch=torch_version, mlx=mlx_version):
                fake_mlx, fake_core = _fake_mlx(mlx_version)
                runtime._validate_runtime.cache_clear()
                try:
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
                        runtime._validate_runtime()
                finally:
                    runtime._validate_runtime.cache_clear()

    def test_validated_runtime_accepts_supported_stable_releases(self):
        for mlx_version in ("0.32.9", "0.33.0", "1.0.0"):
            with self.subTest(mlx=mlx_version):
                fake_mlx, fake_core = _fake_mlx(mlx_version)
                runtime._validate_runtime.cache_clear()
                try:
                    with (
                        mock.patch.dict(
                            sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}
                        ),
                        mock.patch.object(torch, "__version__", "2.13.7"),
                        mock.patch.object(
                            torch.backends.mps, "is_available", return_value=True
                        ),
                    ):
                        self.assertIsNone(runtime._validate_runtime())
                finally:
                    runtime._validate_runtime.cache_clear()

    def test_missing_mlx_has_an_actionable_error(self):
        runtime._validate_runtime.cache_clear()
        try:
            with mock.patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):
                with self.assertRaisesRegex(RuntimeError, "MLX is not installed"):
                    runtime._validate_runtime()
        finally:
            runtime._validate_runtime.cache_clear()

    def test_unavailable_metal_devices_have_actionable_errors(self):
        fake_mlx = types.ModuleType("mlx")
        fake_core = types.ModuleType("mlx.core")
        fake_core.__version__ = "0.32.0"
        fake_mlx.core = fake_core

        cases = (
            (False, True, "PyTorch MPS device"),
            (True, False, "MLX Metal device"),
        )
        for torch_mps_available, mlx_metal_available, message in cases:
            with self.subTest(message=message):
                fake_core.metal = types.SimpleNamespace(
                    is_available=lambda: mlx_metal_available
                )
                runtime._validate_runtime.cache_clear()
                try:
                    with mock.patch.dict(
                        sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}
                    ):
                        with mock.patch.object(torch, "__version__", "2.13.0"):
                            with mock.patch.object(
                                torch.backends.mps,
                                "is_available",
                                return_value=torch_mps_available,
                            ):
                                with self.assertRaisesRegex(RuntimeError, message):
                                    runtime._validate_runtime()
                finally:
                    runtime._validate_runtime.cache_clear()

    @unittest.skipUnless(
        importlib.util.find_spec("mlx") is not None
        and torch.backends.mps.is_available(),
        "requires MLX and MPS",
    )
    def test_incompatible_runtime_aborts_server_args_before_dummy_shortcut(self):
        import mlx.core as mx

        runtime.use_mlx.cache_clear()
        runtime._validate_runtime.cache_clear()
        try:
            with mock.patch.dict(os.environ, {"SGLANG_USE_MLX": "1"}):
                with mock.patch.object(torch, "__version__", "2.12.1"):
                    with self.assertRaisesRegex(RuntimeError, "stable Torch 2.13.x"):
                        from sglang.srt.server_args import ServerArgs

                        ServerArgs(model_path="dummy")

                runtime.use_mlx.cache_clear()
                runtime._validate_runtime.cache_clear()
                with mock.patch.object(mx, "__version__", "0.31.0"):
                    with self.assertRaisesRegex(RuntimeError, "MLX >= 0.32.0"):
                        ServerArgs(model_path="dummy")
        finally:
            runtime.use_mlx.cache_clear()
            runtime._validate_runtime.cache_clear()

    @unittest.skipUnless(
        importlib.util.find_spec("mlx") is not None
        and torch.backends.mps.is_available()
        and not Version(torch.__version__).is_prerelease
        and Version(torch.__version__).release[:2] == (2, 13)
        and _has_supported_mlx(),
        "requires the supported MLX runtime",
    )
    def test_current_runtime_is_supported(self):
        runtime._validate_runtime.cache_clear()
        runtime._validate_runtime()


if __name__ == "__main__":
    unittest.main()
