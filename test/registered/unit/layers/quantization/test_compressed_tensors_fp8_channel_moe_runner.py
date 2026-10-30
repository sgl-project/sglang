"""CPU tests for where compressed-tensors channel-wise FP8 experts run on ROCm.

The experts are preshuffled for AITER's fused MoE only when that runner is used;
the Triton runner reads them unshuffled.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_fp8_moe as scheme_module,
)
from sglang.test.test_utils import CustomTestCase


def _scheme(strategy):
    scheme = scheme_module.CompressedTensorsW8A8Fp8MoE.__new__(
        scheme_module.CompressedTensorsW8A8Fp8MoE
    )
    scheme.weight_quant = SimpleNamespace(strategy=strategy)
    return scheme


class TestChannelExpertsOnAiter(CustomTestCase):
    def _on_aiter(self, runner, use_aiter=True, strategy=QuantizationStrategy.CHANNEL):
        a2a = SimpleNamespace(supports_aiter=lambda: True)
        with (
            mock.patch.object(scheme_module, "_use_aiter", use_aiter),
            mock.patch.object(
                scheme_module, "get_moe_runner_backend", return_value=runner
            ),
            mock.patch.object(scheme_module, "get_moe_a2a_backend", return_value=a2a),
        ):
            return _scheme(strategy)._channel_experts_on_aiter()

    def test_auto_runner_uses_aiter_when_enabled(self):
        self.assertTrue(self._on_aiter(MoeRunnerBackend.AUTO))
        self.assertFalse(self._on_aiter(MoeRunnerBackend.AUTO, use_aiter=False))

    def test_explicit_triton_runner_keeps_weights_unshuffled(self):
        self.assertFalse(self._on_aiter(MoeRunnerBackend.TRITON))

    def test_explicit_aiter_runner(self):
        self.assertTrue(self._on_aiter(MoeRunnerBackend.AITER))

    def test_other_strategies_are_not_channel_wise(self):
        self.assertFalse(
            self._on_aiter(MoeRunnerBackend.AUTO, strategy=QuantizationStrategy.TENSOR)
        )


if __name__ == "__main__":
    unittest.main()
