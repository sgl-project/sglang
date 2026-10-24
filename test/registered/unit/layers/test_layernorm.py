"""Unit tests for srt/layers/layernorm.py"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import layernorm
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 64
NUM_TOKENS = 4


def _make_norm(weight_dtype: torch.dtype = torch.bfloat16, **kwargs) -> RMSNorm:
    norm = RMSNorm(HIDDEN, eps=1e-6, weight_dtype=weight_dtype, **kwargs)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.1 * torch.randn(HIDDEN))
    return norm


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    # The second half of each row is 10x larger, so a partial-row variance
    # (var_hidden_size) differs clearly from the full-row one.
    x = torch.randn(NUM_TOKENS, HIDDEN)
    x[:, HIDDEN // 2 :] *= 10
    return x.bfloat16(), torch.randn(NUM_TOKENS, HIDDEN).bfloat16()


class TestRMSNormHipWithoutVllm(CustomTestCase):
    """forward_hip with neither AITER nor vllm (e.g. RDNA): torch's rms_norm
    where it matches forward_native's semantics, forward_native everywhere else."""

    def setUp(self):
        torch.manual_seed(0)
        patcher = patch.object(layernorm, "_has_vllm_rms_norm", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_plain_case_avoids_forward_native_and_matches_it(self):
        # Rerouting to forward_native is a silent decode regression on RDNA;
        # the residual feeds the next layer, so it must match native exactly.
        norm = _make_norm()
        x, residual = _make_inputs()
        expected = norm.forward_native(x)
        expected_out, expected_residual = norm.forward_native(x, residual)

        with patch.object(
            norm, "forward_native", side_effect=AssertionError("native path taken")
        ):
            out = norm.forward_hip(x)
            out_with_residual, new_residual = norm.forward_hip(x, residual)

        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(out_with_residual, expected_out)
        self.assertEqual(new_residual.dtype, expected_residual.dtype)
        self.assertTrue(torch.equal(new_residual, expected_residual))

    def test_modes_torch_rms_norm_cannot_express_keep_native_output(self):
        # Each mode changes the math or dtypes in ways torch's rms_norm does not
        # reproduce, so forward_hip must return forward_native's exact result.
        cases = {
            "var_hidden_size": dict(norm_kwargs=dict(var_hidden_size=HIDDEN // 2)),
            "cast_x_before_out_mul": dict(norm_kwargs=dict(cast_x_before_out_mul=True)),
            "fp32_residual": dict(norm_kwargs=dict(fp32_residual=True)),
            "override_orig_dtype": dict(
                norm_kwargs=dict(override_orig_dtype=torch.float32)
            ),
            "weight_dtype_mismatch": dict(norm_kwargs=dict(weight_dtype=torch.float32)),
            "post_residual_addition": dict(post_residual_addition=True),
            "batch_invariant_mode": dict(batch_invariant=True),
        }
        for name, case in cases.items():
            with self.subTest(mode=name):
                norm = _make_norm(**case.get("norm_kwargs", {}))
                x, residual = _make_inputs()
                post = (
                    torch.randn_like(residual)
                    if case.get("post_residual_addition")
                    else None
                )
                with patch.object(
                    layernorm,
                    "is_batch_invariant_mode_enabled",
                    return_value=case.get("batch_invariant", False),
                ):
                    out, new_residual = norm.forward_hip(x, residual, post)
                expected_out, expected_residual = norm.forward_native(x, residual, post)
                self.assertEqual(out.dtype, expected_out.dtype)
                self.assertTrue(torch.equal(out, expected_out))
                self.assertEqual(new_residual.dtype, expected_residual.dtype)
                self.assertTrue(torch.equal(new_residual, expected_residual))


if __name__ == "__main__":
    unittest.main()
