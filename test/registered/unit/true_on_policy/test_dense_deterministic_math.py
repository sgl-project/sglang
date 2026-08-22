import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_context
from sglang.srt.true_on_policy import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    get_on_policy_rms_norm_kwargs,
    should_force_bfloat16_dense_tensor_math,
    should_force_bfloat16_lm_head,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="stage-a-test-cpu")


class TestDenseOnPolicyHelpers(unittest.TestCase):
    def test_default_dense_math_helpers_are_inactive(self):
        with get_context().override_server_args(true_on_policy_contract=None):
            self.assertFalse(should_force_bfloat16_dense_tensor_math())
            self.assertFalse(should_force_bfloat16_lm_head(use_fp32_lm_head=False))
            self.assertEqual(get_on_policy_rms_norm_kwargs(), {})

    def test_contract_enables_dense_math_and_rms_norm(self):
        with get_context().override_server_args(
            true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
        ):
            self.assertTrue(should_force_bfloat16_dense_tensor_math())
            self.assertTrue(should_force_bfloat16_lm_head())
            self.assertFalse(should_force_bfloat16_lm_head(use_fp32_lm_head=True))
            self.assertEqual(
                get_on_policy_rms_norm_kwargs(
                    weight_dtype=torch.float32,
                    override_orig_dtype=torch.float32,
                    fp32_residual=True,
                ),
                dict(
                    weight_dtype=torch.float32,
                    override_orig_dtype=torch.float32,
                    cast_x_before_out_mul=True,
                    fp32_residual=True,
                ),
            )


class TestDenseOnPolicyContracts(unittest.TestCase):
    def setUp(self):
        config = get_context().override_server_args(
            true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
        )
        config.install()
        self.addCleanup(config.restore)

    def test_qwen3_style_rms_norm_keeps_fp32_weight_output_and_residual(self):
        from sglang.srt.layers.layernorm import RMSNorm

        norm = RMSNorm(
            4,
            eps=1e-6,
            true_on_policy_weight_dtype=torch.float32,
            true_on_policy_override_orig_dtype=torch.float32,
            true_on_policy_fp32_residual=True,
        )
        x = torch.randn(2, 4, dtype=torch.bfloat16)
        residual = torch.randn(2, 4, dtype=torch.bfloat16)
        out, residual_out = norm.forward_native(x, residual)
        self.assertEqual(norm.weight.dtype, torch.float32)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(residual_out.dtype, torch.float32)

    def test_rms_norm_self_configures_from_role_hints(self):
        from sglang.srt.layers.layernorm import RMSNorm

        norm = RMSNorm(
            4,
            eps=1e-6,
            true_on_policy_weight_dtype=torch.float32,
            true_on_policy_override_orig_dtype=torch.float32,
            true_on_policy_fp32_residual=True,
        )
        self.assertEqual(norm.weight.dtype, torch.float32)
        self.assertTrue(norm.cast_x_before_out_mul)
        self.assertTrue(norm.fp32_residual)
        self.assertEqual(norm.override_orig_dtype, torch.float32)

    def test_lm_head_uses_bfloat16_matmul_inputs(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        head = SimpleNamespace(weight=torch.randn(8, 4, dtype=torch.float32))
        hidden_states = torch.randn(2, 4, dtype=torch.float32)
        with patch("torch.matmul", wraps=torch.matmul) as matmul:
            logits = LogitsProcessor._compute_lm_head(
                SimpleNamespace(use_fp32_lm_head=False),
                hidden_states,
                head,
            )
        a, b = matmul.call_args.args
        self.assertEqual(a.dtype, torch.bfloat16)
        self.assertEqual(b.dtype, torch.bfloat16)
        torch.testing.assert_close(
            logits, hidden_states.bfloat16() @ head.weight.bfloat16().T
        )


if __name__ == "__main__":
    unittest.main()
