"""Unit tests for the FlashInfer CUTLASS MoE SwiGLU parameter materialization."""

import unittest

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    materialize_swiglu_params_for_cutlass,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GPT_OSS_ALPHA = 1.702
GPT_OSS_LIMIT = 7.0
NUM_LOCAL_EXPERTS = 4


class TestMaterializeSwigluParamsForCutlass(CustomTestCase):
    def _materialize(self, activation="silu", **config_kwargs):
        config = MoeRunnerConfig(activation=activation, is_gated=True, **config_kwargs)
        return materialize_swiglu_params_for_cutlass(
            config, num_local_experts=NUM_LOCAL_EXPERTS, device=torch.device("cpu")
        )

    def test_gpt_oss_triple(self):
        # GPT-OSS ships gemm1_alpha=1.702 with a 7.0 clamp and never sets
        # gemm1_beta; alpha implies the +1 up term.
        alpha, beta, limit = self._materialize(
            gemm1_alpha=GPT_OSS_ALPHA, gemm1_clamp_limit=GPT_OSS_LIMIT
        )
        # 1.702 is not exactly representable in fp32; compare with tolerance.
        self.assertTrue(torch.allclose(alpha, torch.full_like(alpha, GPT_OSS_ALPHA)))
        self.assertTrue(torch.allclose(beta, torch.ones_like(beta)))
        self.assertTrue(torch.allclose(limit, torch.full_like(limit, GPT_OSS_LIMIT)))

    def test_gpt_oss_triple_matches_reference_epilogue(self):
        # The reference epilogue is silu(gate) * clamp(alpha*up + 1, ±limit).
        # With the old beta=0.0 default the up term lost its +1.
        torch.manual_seed(0)
        gate = torch.randn(64, dtype=torch.float32)
        up = torch.randn(64, dtype=torch.float32)

        def reference(alpha_v, beta_v, limit_v):
            return torch.nn.functional.silu(gate) * torch.clamp(
                alpha_v * up + beta_v, -limit_v, limit_v
            )

        alpha, beta, limit = self._materialize(
            gemm1_alpha=GPT_OSS_ALPHA, gemm1_clamp_limit=GPT_OSS_LIMIT
        )
        got = reference(alpha[0], beta[0], limit[0])
        want = reference(GPT_OSS_ALPHA, 1.0, GPT_OSS_LIMIT)
        self.assertTrue(torch.allclose(got, want))
        wrong = reference(GPT_OSS_ALPHA, 0.0, GPT_OSS_LIMIT)
        self.assertFalse(torch.allclose(got, wrong))

    def test_explicit_beta_wins(self):
        _, beta, _ = self._materialize(
            gemm1_alpha=GPT_OSS_ALPHA,
            gemm1_beta=0.5,
            gemm1_clamp_limit=GPT_OSS_LIMIT,
        )
        self.assertEqual(beta.tolist(), [0.5] * NUM_LOCAL_EXPERTS)

    def test_no_alpha_stays_neutral(self):
        alpha, beta, _ = self._materialize(gemm1_clamp_limit=GPT_OSS_LIMIT)
        self.assertEqual(alpha.tolist(), [1.0] * NUM_LOCAL_EXPERTS)
        self.assertEqual(beta.tolist(), [0.0] * NUM_LOCAL_EXPERTS)

    def test_swiglu_limit_alias(self):
        _, _, limit = self._materialize(swiglu_limit=GPT_OSS_LIMIT)
        self.assertEqual(limit.tolist(), [GPT_OSS_LIMIT] * NUM_LOCAL_EXPERTS)

    def test_no_clamp_disables(self):
        self.assertEqual(
            self._materialize(gemm1_alpha=GPT_OSS_ALPHA), (None, None, None)
        )

    def test_situ_disables(self):
        # SiTU carries its clamp inside the activation itself.
        self.assertEqual(
            self._materialize(activation="situ", gemm1_clamp_limit=GPT_OSS_LIMIT),
            (None, None, None),
        )

    def test_dtype_and_shape(self):
        for tensor in self._materialize(
            gemm1_alpha=GPT_OSS_ALPHA, gemm1_clamp_limit=GPT_OSS_LIMIT
        ):
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tuple(tensor.shape), (NUM_LOCAL_EXPERTS,))


if __name__ == "__main__":
    unittest.main()
