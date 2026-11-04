"""Tests for the standard Torch-owned MPS runtime contract."""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.arg_groups.validation_hook import validate_mps_model_config
from sglang.srt.hardware_backend.mps import runtime
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


class TestMpsRuntime(CustomTestCase):
    def test_runtime_accepts_torch_213_and_newer(self):
        for version in ("2.13.0", "2.13.7", "2.14.0", "3.0.0"):
            with (
                self.subTest(version=version),
                mock.patch.object(torch, "__version__", version),
                mock.patch.object(
                    torch.backends.mps, "is_available", return_value=True
                ),
                mock.patch.object(torch.mps, "recommended_max_memory", create=True),
                mock.patch.object(torch.mps, "driver_allocated_memory", create=True),
            ):
                self.assertIsNone(runtime.validate_mps_runtime())

    def test_runtime_does_not_require_mlx_or_metal_kernel_apis(self):
        with (
            mock.patch.object(torch, "__version__", "2.13.4"),
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
            mock.patch.object(
                torch.mps, "recommended_max_memory", return_value=8 << 30, create=True
            ),
            mock.patch.object(
                torch.mps, "driver_allocated_memory", return_value=0, create=True
            ),
            mock.patch.object(torch.mps, "compile_shader", None, create=True),
            mock.patch.object(torch.mps, "load_metallib", None, create=True),
            mock.patch.dict("sys.modules", {"mlx": None, "mlx.core": None}),
        ):
            self.assertIsNone(runtime.validate_mps_runtime())

    def test_runtime_rejects_unsupported_torch_and_missing_memory_apis(self):
        for version in ("2.12.1", "2.13.0rc1"):
            with (
                self.subTest(version=version),
                mock.patch.object(torch, "__version__", version),
                self.assertRaisesRegex(RuntimeError, "Torch >= 2.13"),
            ):
                runtime.validate_mps_runtime()

        with (
            mock.patch.object(torch, "__version__", "2.13.0"),
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
            mock.patch.object(torch.mps, "recommended_max_memory", None, create=True),
            self.assertRaisesRegex(RuntimeError, "recommended_max_memory"),
        ):
            runtime.validate_mps_runtime()

    def test_runtime_rejects_unavailable_mps(self):
        with (
            mock.patch.object(torch, "__version__", "2.13.0"),
            mock.patch.object(torch.backends.mps, "is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "available PyTorch MPS device"),
        ):
            runtime.validate_mps_runtime()

    @staticmethod
    def _resolve_with_gate(device, *, mlx, detected_mps=False):
        """Resolve dummy arguments and return the patched runtime validator."""
        from sglang.srt import server_args
        from sglang.srt.server_args import ServerArgs

        with (
            mock.patch.object(server_args, "use_mlx", return_value=mlx),
            override_platform(is_mps=detected_mps),
            mock.patch.object(server_args, "validate_mps_runtime") as validate,
        ):
            ServerArgs(model_path="dummy", device=device).resolve_once()
        return validate

    def test_server_args_selects_runtime_gate_by_execution_path(self):
        self._resolve_with_gate("mps", mlx=False).assert_called_once_with()
        self._resolve_with_gate("mps", mlx=True).assert_not_called()

    def test_runtime_gate_follows_autodetected_platform_when_device_is_unset(self):
        """Use platform detection when --device is omitted."""
        self._resolve_with_gate(
            None, mlx=False, detected_mps=True
        ).assert_called_once_with()
        self._resolve_with_gate(None, mlx=False, detected_mps=False).assert_not_called()

    def test_checkpoint_derived_execution_modes_are_rejected(self):
        self.assertIsNone(
            validate_mps_model_config(
                types.SimpleNamespace(quantization=None, is_multimodal=False)
            )
        )
        with self.assertRaisesRegex(ValueError, "quantization='awq'"):
            validate_mps_model_config(
                types.SimpleNamespace(quantization="awq", is_multimodal=False)
            )
        with self.assertRaisesRegex(ValueError, "multimodal serving"):
            validate_mps_model_config(
                types.SimpleNamespace(quantization=None, is_multimodal=True)
            )

    def test_standard_path_rejects_unsupported_execution_modes(self):
        from sglang.srt.arg_groups.validation_hook import (
            validate_standard_mps_server_args,
        )

        def make(**overrides):
            base = dict(
                attention_backend="torch_native",
                prefill_attention_backend=None,
                decode_attention_backend=None,
                sampling_backend="pytorch",
                tp_size=1,
                pp_size=1,
                dp_size=1,
                quantization=None,
            )
            base.update(overrides)
            return types.SimpleNamespace(**base)

        self.assertIsNone(validate_standard_mps_server_args(make()))

        for overrides, expected in (
            ({"attention_backend": "fa3"}, "torch_native attention backend"),
            ({"sampling_backend": "flashinfer"}, "pytorch sampling backend"),
            ({"tp_size": 2}, "tp_size=1"),
            ({"pp_size": 2}, "tp_size=1"),
            ({"dp_size": 2}, "tp_size=1"),
        ):
            with self.subTest(**overrides):
                with self.assertRaisesRegex(ValueError, expected):
                    validate_standard_mps_server_args(make(**overrides))


if __name__ == "__main__":
    unittest.main()
