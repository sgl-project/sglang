"""CPU unit tests for Ascend MoE runner activation selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Initialize the quantization registry before importing the Ascend runner.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.layers.moe.moe_runner import ascend as ascend_runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAscendMoERunner(unittest.TestCase):
    def test_unquantized_swiglu_limit_across_routing_backends(self):
        layer = SimpleNamespace(
            w13_kernel=ascend_runner.NPUUnquantMoEMethod(),
            w2_kernel=ascend_runner.NPUUnquantMoEMethod(),
        )
        for backend, default_activation in (
            (MoeA2ABackend.NONE, ascend_runner.NPUSwiglu),
            (MoeA2ABackend.DEEPEP, ascend_runner.NPUSwigluDeepEPKernel),
            (MoeA2ABackend.ASCEND_OPS, ascend_runner.NPUSwigluDeepEPKernel),
        ):
            with (
                self.subTest(backend=backend),
                patch.object(
                    ascend_runner, "get_moe_a2a_backend", return_value=backend
                ),
                # The default A2A activation loads an NPU operator at construction.
                patch.object(
                    ascend_runner.NPUSwigluDeepEPKernel, "__init__", return_value=None
                ),
            ):
                config = MoeRunnerConfig(layer=layer, swiglu_limit=7.0)
                runner = ascend_runner.AscendRunnerCore(config)
                self.assertIsInstance(
                    runner.activation, ascend_runner.NPUSwigluWithLimit
                )
                self.assertEqual(runner.activation.limit, 7.0)

                runner = ascend_runner.AscendRunnerCore(MoeRunnerConfig(layer=layer))
                self.assertIsInstance(runner.activation, default_activation)


if __name__ == "__main__":
    unittest.main()
