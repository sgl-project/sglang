"""CPU unit tests for NVFP4 fused-MoE GEMM1 global-scale handling.

These cover the split that gives the gate (w1) and up (w3) halves of the fused
w13 GEMM their own NVFP4 weight scales, plus the derived TRT-LLM up-half output
scalar (g1_scale_c). The full TRT-LLM path only runs on Blackwell, and most
checkpoints ship near-equal gate/up scales, so a regression here wouldn't show
up in an accuracy eval. Pin the contract directly instead.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

# Import modelopt_quant before flashinfer_trtllm. Importing flashinfer_trtllm
# first hits a pre-existing circular import through the compressed_tensors
# schemes package; the quantization-package-first order masks it. The isort
# guards keep that order.
# isort: off
from sglang.srt.layers.quantization.modelopt_quant import _compute_gemm1_alphas
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import _compute_g1_scale_c

# isort: on
from sglang.test.test_utils import CustomTestCase


class TestGemm1Alphas(CustomTestCase):
    """w13 gate/up global-scale split feeding g1_alphas / g1_alphas_up."""

    def test_distinct_gate_up_columns(self):
        # [num_experts, 2] checkpoint: col 0 = gate, col 1 = up. Each alpha must
        # come from its own column, and the two must not be conflated.
        gate_col = torch.tensor([2.0, 4.0, 8.0])
        up_col = torch.tensor([3.0, 9.0, 27.0])
        w13_weight_scale_2 = torch.stack([gate_col, up_col], dim=1)
        w13_input_scale = torch.tensor(0.5)

        g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
            w13_weight_scale_2, w13_input_scale, is_gated=True
        )

        torch.testing.assert_close(g1_alphas, w13_input_scale * gate_col)
        torch.testing.assert_close(g1_alphas_up, w13_input_scale * up_col)
        self.assertFalse(torch.allclose(g1_alphas, g1_alphas_up))

    def test_columns_are_decoupled(self):
        # Perturbing only the up column must leave the gate alpha untouched (and
        # vice versa). This catches a column swap or accidental coupling that a
        # formula-mirroring assertion would miss.
        gate_col = torch.tensor([2.0, 4.0, 8.0])
        up_col = torch.tensor([3.0, 9.0, 27.0])
        w13_input_scale = torch.tensor(0.5)

        base = torch.stack([gate_col, up_col], dim=1)
        bumped_up = torch.stack([gate_col, up_col * 10], dim=1)
        bumped_gate = torch.stack([gate_col * 10, up_col], dim=1)

        g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
            base, w13_input_scale, is_gated=True
        )
        g1_alphas_bu, g1_alphas_up_bu = _compute_gemm1_alphas(
            bumped_up, w13_input_scale, is_gated=True
        )
        g1_alphas_bg, g1_alphas_up_bg = _compute_gemm1_alphas(
            bumped_gate, w13_input_scale, is_gated=True
        )

        # Bumping up leaves gate alone; bumping gate leaves up alone.
        torch.testing.assert_close(g1_alphas, g1_alphas_bu)
        self.assertFalse(torch.allclose(g1_alphas_up, g1_alphas_up_bu))
        torch.testing.assert_close(g1_alphas_up, g1_alphas_up_bg)
        self.assertFalse(torch.allclose(g1_alphas, g1_alphas_bg))

    def test_alphas_reconstruct_each_gemm1_half(self):
        # What the alphas are *for*: turning a quantized GEMM1 accumulator back
        # into the real (high-precision) output, per half. Simulate a GEMM in
        # the quantized domain (block scales omitted — only the global-scale
        # split is under test) and check each half reconstructs with its own
        # alpha. Using the gate alpha for the up half would mis-scale it.
        torch.manual_seed(0)
        num_experts, m, k, n = 2, 3, 4, 5
        a = torch.randn(num_experts, m, k)
        w_gate = torch.randn(num_experts, n, k)
        w_up = torch.randn(num_experts, n, k)

        a_scale = torch.tensor(0.5)
        gate_wscale = torch.tensor([2.0, 3.0])
        up_wscale = torch.tensor([5.0, 7.0])
        w13_weight_scale_2 = torch.stack([gate_wscale, up_wscale], dim=1)

        a_q = a / a_scale
        acc_gate = torch.einsum(
            "emk,enk->emn", a_q, w_gate / gate_wscale[:, None, None]
        )
        acc_up = torch.einsum("emk,enk->emn", a_q, w_up / up_wscale[:, None, None])

        g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
            w13_weight_scale_2, a_scale, is_gated=True
        )

        ref_gate = torch.einsum("emk,enk->emn", a, w_gate)
        ref_up = torch.einsum("emk,enk->emn", a, w_up)
        torch.testing.assert_close(acc_gate * g1_alphas[:, None, None], ref_gate)
        torch.testing.assert_close(acc_up * g1_alphas_up[:, None, None], ref_up)

    def test_shared_scale_forms_are_equivalent(self):
        # A shared w1/w3 scale can arrive as 1-D, [num_experts, 1], or
        # [num_experts, 2] with equal columns. All three must produce the same
        # result with gate == up, and the single-column form must not index the
        # missing column 1.
        w13_input_scale = torch.tensor(0.5)
        shared = torch.tensor([2.0, 4.0, 8.0])
        expected = w13_input_scale * shared
        forms = {
            "1d": shared,
            "single_column": shared.unsqueeze(1),
            "equal_columns": torch.stack([shared, shared], dim=1),
        }

        for name, w13_weight_scale_2 in forms.items():
            with self.subTest(form=name):
                g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
                    w13_weight_scale_2, w13_input_scale, is_gated=True
                )
                torch.testing.assert_close(g1_alphas, expected)
                torch.testing.assert_close(g1_alphas_up, expected)

    def test_non_gated_shares_scale(self):
        # Non-gated layers carry a 1-D scale and never split.
        scale = torch.tensor([2.0, 4.0, 8.0])
        w13_input_scale = torch.tensor(0.5)

        g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
            scale, w13_input_scale, is_gated=False
        )

        torch.testing.assert_close(g1_alphas, w13_input_scale * scale)
        torch.testing.assert_close(g1_alphas, g1_alphas_up)

    def test_alphas_are_float32(self):
        # Downstream kernels read these as fp32 regardless of checkpoint dtype.
        w13_weight_scale_2 = torch.ones(2, 2, dtype=torch.float16)
        w13_input_scale = torch.tensor(1.0, dtype=torch.float16)

        g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
            w13_weight_scale_2, w13_input_scale, is_gated=True
        )

        self.assertEqual(g1_alphas.dtype, torch.float32)
        self.assertEqual(g1_alphas_up.dtype, torch.float32)


class TestG1ScaleC(CustomTestCase):
    """TRT-LLM up-half output scalar derived from g1_alphas_up."""

    def test_gated_tracks_up_and_ignores_gate(self):
        # g1_scale_c dequantizes the up half, so it must equal w2_input_scale_quant
        # * g1_alphas_up and be independent of the gate alpha (the pre-split bug
        # used the gate alpha here).
        w2_input_scale_quant = torch.tensor([0.5, 0.25, 0.1])
        gate = torch.tensor([6.0, 12.0, 24.0])
        up = torch.tensor([9.0, 36.0, 108.0])

        g1_scale_c = _compute_g1_scale_c(w2_input_scale_quant, gate, up, is_gated=True)
        torch.testing.assert_close(g1_scale_c, w2_input_scale_quant * up)

        # Changing only the gate alpha must not move the result.
        g1_scale_c_other_gate = _compute_g1_scale_c(
            w2_input_scale_quant, gate * 5, up, is_gated=True
        )
        torch.testing.assert_close(g1_scale_c, g1_scale_c_other_gate)

    def test_gated_reduces_to_single_scale_when_up_equals_gate(self):
        # Shared-scale checkpoints pass g1_alphas in as g1_alphas_up, which
        # matches the original single-scale path.
        w2_input_scale_quant = torch.tensor([0.5, 0.25])
        g1_alphas = torch.tensor([6.0, 12.0])

        g1_scale_c = _compute_g1_scale_c(
            w2_input_scale_quant, g1_alphas, g1_alphas, is_gated=True
        )

        torch.testing.assert_close(g1_scale_c, w2_input_scale_quant * g1_alphas)

    def test_non_gated_is_reciprocal_a2_per_expert(self):
        # Relu2-style: no gate dequant contribution, just 1/a2_scale broadcast
        # to one value per expert. The alphas are ignored apart from the count.
        w2_input_scale_quant = torch.tensor(0.5)
        g1_alphas = torch.zeros(4)

        g1_scale_c = _compute_g1_scale_c(
            w2_input_scale_quant, g1_alphas, g1_alphas, is_gated=False
        )

        self.assertEqual(g1_scale_c.shape, (4,))
        torch.testing.assert_close(g1_scale_c, torch.full((4,), 0.5))

    def test_scale_c_is_float32(self):
        w2_input_scale_quant = torch.ones(2, dtype=torch.float16)
        g1_alphas_up = torch.ones(2, dtype=torch.float16)

        g1_scale_c = _compute_g1_scale_c(
            w2_input_scale_quant, g1_alphas_up, g1_alphas_up, is_gated=True
        )

        self.assertEqual(g1_scale_c.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
