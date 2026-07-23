"""CPU unit tests for NVFP4 MoE automatic backend selection."""

import unittest
from unittest.mock import patch

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
    _resolve_nvfp4_moe_runner_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelOptNvfp4MoeBackend(CustomTestCase):
    def _resolve_auto_backend(self, capability):
        with (
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.get_device_capability",
                return_value=capability,
            ),
        ):
            return _resolve_nvfp4_moe_runner_backend(MoeRunnerBackend.AUTO)

    def test_auto_uses_flashinfer_cutlass_on_sm100_and_newer(self):
        for capability in ((10, 0), (10, 3), (12, 0)):
            with self.subTest(capability=capability):
                self.assertEqual(
                    self._resolve_auto_backend(capability),
                    MoeRunnerBackend.FLASHINFER_CUTLASS,
                )

    def test_auto_keeps_marlin_fallback_below_sm100(self):
        self.assertEqual(
            self._resolve_auto_backend((9, 0)),
            MoeRunnerBackend.MARLIN,
        )

    def test_auto_requires_cuda(self):
        with patch(
            "sglang.srt.layers.quantization.modelopt_quant.is_cuda",
            return_value=False,
        ):
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                _resolve_nvfp4_moe_runner_backend(MoeRunnerBackend.AUTO)

    def test_explicit_backend_is_unchanged(self):
        for backend in (
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            MoeRunnerBackend.FLASHINFER_CUTLASS,
            MoeRunnerBackend.MARLIN,
        ):
            with self.subTest(backend=backend):
                self.assertIs(
                    _resolve_nvfp4_moe_runner_backend(backend),
                    backend,
                )

    def test_cutlass_property_uses_effective_backend(self):
        method = object.__new__(ModelOptNvFp4FusedMoEMethod)
        method._moe_runner_backend = MoeRunnerBackend.FLASHINFER_CUTLASS

        with patch(
            "sglang.srt.layers.moe.get_moe_runner_backend",
            return_value=MoeRunnerBackend.AUTO,
        ):
            self.assertTrue(method.enable_flashinfer_cutlass_moe)


if __name__ == "__main__":
    unittest.main()
