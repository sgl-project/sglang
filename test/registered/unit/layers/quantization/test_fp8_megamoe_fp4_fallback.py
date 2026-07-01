"""Unit tests for the FP4 MegaMoE fallback runner selection.

MegaMoE uses its specialized kernels while the token count stays under the
configured cap. When a request exceeds that cap, the model falls back to the
normal MoE runner created here. For DeepSeek-V4 FP4 experts, that fallback must
use DeepGEMM because the MegaMoE weight preparation produces the scale layout
that DeepGEMM consumes; Triton's FP8 path rejects it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod


class TestFp8MegaMoeFp4Fallback(unittest.TestCase):
    def _create_runner_backend(
        self,
        *,
        is_fp4_experts,
        a2a_backend,
        runner_backend=MoeRunnerBackend.AUTO,
        deepgemm_enabled=False,
    ):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[1, 32],
            is_fp4_experts=is_fp4_experts,
        )

        with patch(
            "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
            return_value=runner_backend,
        ):
            method = Fp8MoEMethod(quant_config)

        with patch(
            "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
            return_value=runner_backend,
        ), patch(
            "sglang.srt.layers.quantization.fp8.get_moe_a2a_backend",
            return_value=a2a_backend,
        ), patch.object(
            Fp8MoEMethod,
            "is_deepgemm_moe_runner_backend_enabled",
            return_value=deepgemm_enabled,
        ), patch(
            "sglang.srt.layers.quantization.fp8.MoeRunner"
        ) as mock_runner:
            method.create_moe_runner(object(), MoeRunnerConfig())

        mock_runner.assert_called_once()
        return mock_runner.call_args.args[0]

    def test_auto_megamoe_fp4_experts_use_deep_gemm(self):
        backend = self._create_runner_backend(
            is_fp4_experts=True,
            a2a_backend=MoeA2ABackend.MEGAMOE,
        )

        self.assertEqual(backend, MoeRunnerBackend.DEEP_GEMM)

    def test_auto_megamoe_regular_fp8_keeps_triton_default(self):
        backend = self._create_runner_backend(
            is_fp4_experts=False,
            a2a_backend=MoeA2ABackend.MEGAMOE,
        )

        self.assertEqual(backend, MoeRunnerBackend.TRITON)

    def test_auto_non_megamoe_fp4_keeps_triton_default(self):
        backend = self._create_runner_backend(
            is_fp4_experts=True,
            a2a_backend=MoeA2ABackend.NONE,
        )

        self.assertEqual(backend, MoeRunnerBackend.TRITON)

    def test_auto_existing_deep_gemm_detection_is_preserved(self):
        backend = self._create_runner_backend(
            is_fp4_experts=False,
            a2a_backend=MoeA2ABackend.DEEPEP,
            deepgemm_enabled=True,
        )

        self.assertEqual(backend, MoeRunnerBackend.DEEP_GEMM)

    def test_explicit_triton_is_preserved(self):
        backend = self._create_runner_backend(
            is_fp4_experts=True,
            a2a_backend=MoeA2ABackend.MEGAMOE,
            runner_backend=MoeRunnerBackend.TRITON,
        )

        self.assertEqual(backend, MoeRunnerBackend.TRITON)


if __name__ == "__main__":
    unittest.main(verbosity=3)
