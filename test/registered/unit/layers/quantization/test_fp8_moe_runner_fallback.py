"""Fp8MoEMethod builds a triton runner when the global MoE runner backend is
flashinfer_cutlass or flashinfer_cutedsl, which have no fp8 MoE path."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import sglang.srt.layers.quantization.fp8 as fp8
from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
from sglang.test.test_utils import CustomTestCase


class TestFp8MoeRunnerFallback(CustomTestCase):
    def _runner_backend_for(self, global_backend):
        method = fp8.Fp8MoEMethod.__new__(fp8.Fp8MoEMethod)
        with patch.object(fp8, "get_moe_runner_backend", return_value=global_backend):
            method.create_moe_runner(layer=None, moe_runner_config=MoeRunnerConfig())
        return method.runner.runner_backend

    def test_flashinfer_cutlass_falls_back_to_triton(self):
        self.assertTrue(
            self._runner_backend_for(MoeRunnerBackend.FLASHINFER_CUTLASS).is_triton()
        )

    def test_flashinfer_cutedsl_falls_back_to_triton(self):
        self.assertTrue(
            self._runner_backend_for(MoeRunnerBackend.FLASHINFER_CUTEDSL).is_triton()
        )

    def test_triton_is_kept(self):
        self.assertTrue(self._runner_backend_for(MoeRunnerBackend.TRITON).is_triton())


if __name__ == "__main__":
    unittest.main()
