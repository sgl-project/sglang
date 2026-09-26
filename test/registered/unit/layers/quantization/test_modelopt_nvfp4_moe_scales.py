"""CPU unit tests for NVFP4 fused-MoE GEMM1 global-scale handling.

w13 fuses the gate (w1) and up (w3) projections, which can carry separate NVFP4
weight scales. These tests pin the two helpers that handle that: the gate/up
split in _compute_gemm1_alphas and the TRT-LLM up-half output scalar in
_compute_g1_scale_c. Both are plain tensor ops, so they run on CPU.

The full TRT-LLM path only runs on Blackwell, and most checkpoints ship
near-equal gate/up scales, so this regression wouldn't surface in an accuracy
eval -- hence the direct contract tests. The wiring around the helpers (Marlin's
single-scale collapse, the w1/w3 mismatch warning, and registering g1_alphas /
g1_alphas_up in process_weights_after_loading) needs the GPU kernels and is
covered on-device.

Shapes and scale magnitudes follow real NVFP4 MoE checkpoints rather than toy
sizes, so a failure looks like one a real checkpoint would hit.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest

import torch

# Import modelopt_quant before flashinfer_trtllm. Importing flashinfer_trtllm
# first hits a pre-existing circular import through the compressed_tensors
# schemes package; the quantization-package-first order masks it. The isort
# guards keep that order.
# isort: off
from sglang.srt.layers.quantization.modelopt_quant import _compute_gemm1_alphas
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    _align_fp4_moe_weights,
    _compute_g1_scale_c,
    trtllm_nvfp4_hidden_alignment,
)

# isort: on
from sglang.test.test_utils import CustomTestCase

# Representative NVFP4 MoE shapes (name, num_experts, hidden, intermediate,
# is_gated). Only num_experts and is_gated matter to the scale helpers; hidden
# and intermediate just feed the reconstruct GEMM.
REAL_MOE_CONFIGS = [
    ("deepseek-r1", 256, 7168, 2048, True),
    ("llama4-maverick", 128, 5120, 8192, True),
    ("qwen3-235b", 128, 4096, 1536, True),
    ("mixtral-8x7b", 8, 4096, 14336, True),
    ("relu2-nongated", 128, 4096, 4096, False),
]
GATED_CONFIGS = [c for c in REAL_MOE_CONFIGS if c[-1]]
NONGATED_CONFIG = next(c for c in REAL_MOE_CONFIGS if not c[-1])


def _global_scales(num_experts: int, num_cols: int, seed: int) -> torch.Tensor:
    """Small positive fp32 weight scales, distinct per expert and column.

    Real NVFP4 global scales are amax/(448*6)-style values, ~O(1e-2). Keeping
    them distinct makes a gate/up swap visible. A local generator avoids
    touching the global RNG.
    """
    g = torch.Generator().manual_seed(seed)
    scales = torch.rand((num_experts, num_cols), generator=g) * 0.19 + 0.01
    return scales if num_cols > 1 else scales.squeeze(1)


class TestGemm1Alphas(CustomTestCase):
    """w13 gate/up global-scale split feeding g1_alphas / g1_alphas_up."""

    def test_distinct_gate_up_columns(self):
        # col 0 is gate, col 1 is up; each alpha must read its own column.
        w13_input_scale = torch.tensor(0.05)
        for i, (name, num_experts, _, _, _) in enumerate(GATED_CONFIGS):
            with self.subTest(config=name):
                w13_weight_scale_2 = _global_scales(num_experts, 2, seed=i)
                gate_col = w13_weight_scale_2[:, 0]
                up_col = w13_weight_scale_2[:, 1]

                g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
                    w13_weight_scale_2, w13_input_scale, is_gated=True
                )

                self.assertEqual(g1_alphas.shape, (num_experts,))
                self.assertEqual(g1_alphas_up.shape, (num_experts,))
                torch.testing.assert_close(g1_alphas, w13_input_scale * gate_col)
                torch.testing.assert_close(g1_alphas_up, w13_input_scale * up_col)
                self.assertFalse(torch.allclose(g1_alphas, g1_alphas_up))

    def test_columns_are_decoupled(self):
        # Bump one column and the other alpha must not move. Catches a column
        # swap or coupling that a formula-mirroring check would wave through.
        w13_input_scale = torch.tensor(0.05)
        for i, (name, num_experts, _, _, _) in enumerate(GATED_CONFIGS):
            with self.subTest(config=name):
                base = _global_scales(num_experts, 2, seed=i)
                gate_col, up_col = base[:, 0], base[:, 1]
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

                torch.testing.assert_close(g1_alphas, g1_alphas_bu)
                self.assertFalse(torch.allclose(g1_alphas_up, g1_alphas_up_bu))
                torch.testing.assert_close(g1_alphas_up, g1_alphas_up_bg)
                self.assertFalse(torch.allclose(g1_alphas, g1_alphas_bg))

    def test_per_expert_input_scale(self):
        # The non-TRT-LLM backends pass a per-expert w13_input_scale (rather than
        # the scalar the TRT-LLM path uses); it should scale each column
        # element-wise.
        num_experts = GATED_CONFIGS[0][1]
        w13_weight_scale_2 = _global_scales(num_experts, 2, seed=0)
        w13_input_scale = _global_scales(num_experts, 1, seed=5)  # per-expert [E]

        g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
            w13_weight_scale_2, w13_input_scale, is_gated=True
        )

        torch.testing.assert_close(
            g1_alphas, w13_input_scale * w13_weight_scale_2[:, 0]
        )
        torch.testing.assert_close(
            g1_alphas_up, w13_input_scale * w13_weight_scale_2[:, 1]
        )

    def test_alphas_reconstruct_each_gemm1_half(self):
        # The alphas dequantize the GEMM1 accumulator back to full precision, one
        # per half. Fake a GEMM in the quantized domain (block scales omitted --
        # only the global-scale split is under test) and check each half
        # reconstructs with its own alpha; the gate alpha would mis-scale the up
        # half. Real expert counts, but k/n are capped (the reconstruction math
        # doesn't depend on them) and tokens kept small to fit CPU CI. atol
        # covers fp32 accumulation, not the split -- a wrong alpha is off by an
        # O(1) factor, far above it.
        m, k_cap, n_cap = 8, 128, 128
        a_scale = torch.tensor(0.05)
        for i, (name, num_experts, hidden, inter, _) in enumerate(GATED_CONFIGS):
            with self.subTest(config=name):
                k, n = min(hidden, k_cap), min(inter, n_cap)
                gen = torch.Generator().manual_seed(100 + i)
                a = torch.randn(num_experts, m, k, generator=gen)
                w_gate = torch.randn(num_experts, n, k, generator=gen)
                w_up = torch.randn(num_experts, n, k, generator=gen)

                w13_weight_scale_2 = _global_scales(num_experts, 2, seed=i)
                gate_wscale = w13_weight_scale_2[:, 0]
                up_wscale = w13_weight_scale_2[:, 1]

                a_q = a / a_scale
                acc_gate = torch.einsum(
                    "emk,enk->emn", a_q, w_gate / gate_wscale[:, None, None]
                )
                acc_up = torch.einsum(
                    "emk,enk->emn", a_q, w_up / up_wscale[:, None, None]
                )

                g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
                    w13_weight_scale_2, a_scale, is_gated=True
                )

                ref_gate = torch.einsum("emk,enk->emn", a, w_gate)
                ref_up = torch.einsum("emk,enk->emn", a, w_up)
                torch.testing.assert_close(
                    acc_gate * g1_alphas[:, None, None], ref_gate, rtol=1e-3, atol=1e-3
                )
                torch.testing.assert_close(
                    acc_up * g1_alphas_up[:, None, None], ref_up, rtol=1e-3, atol=1e-3
                )

    def test_shared_scale_forms_are_equivalent(self):
        # A shared w1/w3 scale can arrive 1-D, [E, 1], or [E, 2] with equal
        # columns. All three should give gate == up, and the single-column form
        # must not reach for the missing column 1.
        w13_input_scale = torch.tensor(0.05)
        for i, (name, num_experts, _, _, _) in enumerate(GATED_CONFIGS):
            shared = _global_scales(num_experts, 1, seed=i)
            expected = w13_input_scale * shared
            forms = {
                "1d": shared,
                "single_column": shared.unsqueeze(1),
                "equal_columns": torch.stack([shared, shared], dim=1),
            }
            for form_name, w13_weight_scale_2 in forms.items():
                with self.subTest(config=name, form=form_name):
                    g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
                        w13_weight_scale_2, w13_input_scale, is_gated=True
                    )
                    torch.testing.assert_close(g1_alphas, expected)
                    torch.testing.assert_close(g1_alphas_up, expected)

    def test_non_gated_shares_scale(self):
        # Non-gated layers carry one 1-D scale and never split.
        _, num_experts, _, _, _ = NONGATED_CONFIG
        scale = _global_scales(num_experts, 1, seed=0)
        w13_input_scale = torch.tensor(0.05)

        g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
            scale, w13_input_scale, is_gated=False
        )

        self.assertEqual(g1_alphas.shape, (num_experts,))
        torch.testing.assert_close(g1_alphas, w13_input_scale * scale)
        torch.testing.assert_close(g1_alphas, g1_alphas_up)

    def test_alphas_are_float32(self):
        # Checkpoints store the scale as fp32 and kernels read fp32 alphas; a
        # lower-precision input is still upcast.
        num_experts = GATED_CONFIGS[0][1]
        fp32_scale = _global_scales(num_experts, 2, seed=0)
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                w13_weight_scale_2 = fp32_scale.to(dtype)
                w13_input_scale = torch.tensor(0.05, dtype=dtype)

                g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
                    w13_weight_scale_2, w13_input_scale, is_gated=True
                )

                self.assertEqual(g1_alphas.dtype, torch.float32)
                self.assertEqual(g1_alphas_up.dtype, torch.float32)


class TestG1ScaleC(CustomTestCase):
    """TRT-LLM up-half output scalar derived from g1_alphas_up."""

    def test_gated_tracks_up_and_ignores_gate(self):
        # g1_scale_c scales the up half, so it tracks g1_alphas_up and ignores the
        # gate alpha (the old bug used the gate alpha here). The real path passes a
        # scalar w2_input_scale_quant = 1 / w2_input_scale.max(), which broadcasts
        # to one value per expert.
        w2_input_scale_quant = torch.tensor(20.0)  # ~ 1 / 0.05, scalar
        for i, (name, num_experts, _, _, _) in enumerate(GATED_CONFIGS):
            with self.subTest(config=name):
                gate = _global_scales(num_experts, 1, seed=i)
                up = _global_scales(num_experts, 1, seed=100 + i)

                g1_scale_c = _compute_g1_scale_c(
                    w2_input_scale_quant, gate, up, is_gated=True
                )
                self.assertEqual(g1_scale_c.shape, (num_experts,))
                torch.testing.assert_close(g1_scale_c, w2_input_scale_quant * up)

                # Moving only the gate alpha must not move the result.
                g1_scale_c_other_gate = _compute_g1_scale_c(
                    w2_input_scale_quant, gate * 5, up, is_gated=True
                )
                torch.testing.assert_close(g1_scale_c, g1_scale_c_other_gate)

    def test_gated_accepts_per_expert_a2_scale(self):
        # A per-expert (not scalar) w2_input_scale_quant should still broadcast.
        num_experts = GATED_CONFIGS[0][1]
        w2_input_scale_quant = _global_scales(num_experts, 1, seed=7)
        gate = _global_scales(num_experts, 1, seed=8)
        up = _global_scales(num_experts, 1, seed=9)

        g1_scale_c = _compute_g1_scale_c(w2_input_scale_quant, gate, up, is_gated=True)

        self.assertEqual(g1_scale_c.shape, (num_experts,))
        torch.testing.assert_close(g1_scale_c, w2_input_scale_quant * up)

    def test_gated_reduces_to_single_scale_when_up_equals_gate(self):
        # Shared-scale checkpoints pass g1_alphas as g1_alphas_up, recovering the
        # old single-scale value.
        num_experts = GATED_CONFIGS[0][1]
        w2_input_scale_quant = torch.tensor(20.0)
        g1_alphas = _global_scales(num_experts, 1, seed=3)

        g1_scale_c = _compute_g1_scale_c(
            w2_input_scale_quant, g1_alphas, g1_alphas, is_gated=True
        )

        torch.testing.assert_close(g1_scale_c, w2_input_scale_quant * g1_alphas)

    def test_non_gated_is_reciprocal_a2_per_expert(self):
        # Relu2: no gate dequant, just 1/a2_scale as one contiguous value per
        # expert. The alphas are ignored apart from the count.
        _, num_experts, _, _, _ = NONGATED_CONFIG
        w2_input_scale_quant = torch.tensor(20.0)
        g1_alphas = torch.zeros(num_experts)

        g1_scale_c = _compute_g1_scale_c(
            w2_input_scale_quant, g1_alphas, g1_alphas, is_gated=False
        )

        self.assertEqual(g1_scale_c.shape, (num_experts,))
        self.assertTrue(g1_scale_c.is_contiguous())
        torch.testing.assert_close(g1_scale_c, torch.full((num_experts,), 20.0))

    def test_situ_keeps_both_dequant_scales_inside_activation(self):
        # SiTU keeps both GEMM1 scales before tanh; scale_c contains only the
        # GEMM2 input requantization.
        num_experts = GATED_CONFIGS[0][1]
        w2_input_scale_quant = torch.tensor(20.0)
        gate = _global_scales(num_experts, 1, seed=8)
        up = _global_scales(num_experts, 1, seed=9)

        g1_scale_c = _compute_g1_scale_c(
            w2_input_scale_quant,
            gate,
            up,
            is_gated=True,
            activation="situ",
        )

        self.assertEqual(g1_scale_c.shape, (num_experts,))
        self.assertTrue(g1_scale_c.is_contiguous())
        torch.testing.assert_close(g1_scale_c, torch.full((num_experts,), 20.0))

    def test_scale_c_is_float32(self):
        # Lower-precision inputs are upcast to fp32 for the kernel.
        num_experts = GATED_CONFIGS[0][1]
        w2_input_scale_quant = torch.ones(num_experts, dtype=torch.float16)
        g1_alphas_up = torch.ones(num_experts, dtype=torch.float16)

        g1_scale_c = _compute_g1_scale_c(
            w2_input_scale_quant, g1_alphas_up, g1_alphas_up, is_gated=True
        )

        self.assertEqual(g1_scale_c.dtype, torch.float32)


def _fp4_moe_weights(num_experts: int, hidden: int, inter: int, is_gated: bool):
    """Packed FP4 weights + E4M3 block scales in checkpoint layout, no zeros."""
    gate_up = (2 if is_gated else 1) * inter
    u8 = lambda *shape: torch.randint(1, 255, shape, dtype=torch.uint8)
    w13 = u8(num_experts, gate_up, hidden // 2)
    w13_scale = u8(num_experts, gate_up, hidden // 16).view(torch.float8_e4m3fn)
    w2 = u8(num_experts, hidden, inter // 2)
    w2_scale = u8(num_experts, hidden, inter // 16).view(torch.float8_e4m3fn)
    return w13, w13_scale, w2, w2_scale


class TestAlignFp4MoeWeights(CustomTestCase):
    """Intermediate/hidden zero-padding for the TRT-LLM NVFP4 MoE weights.

    Nemotron 3 / 3.5 (hidden 2688, intermediate 1856, RELU^2 experts) needs
    both: the non-gated kernel wants a 128-aligned intermediate, and the
    per-token variant only has a usable tactic set at a 512-aligned hidden.
    """

    def test_hidden_alignment_only_for_per_token_non_gated(self):
        self.assertEqual(trtllm_nvfp4_hidden_alignment(True, False), 512)
        self.assertEqual(trtllm_nvfp4_hidden_alignment(True, True), 1)
        self.assertEqual(trtllm_nvfp4_hidden_alignment(False, False), 1)
        self.assertEqual(trtllm_nvfp4_hidden_alignment(False, True), 1)

    def test_aligned_shapes_are_returned_untouched(self):
        weights = _fp4_moe_weights(4, 3072, 1920, is_gated=False)

        out = _align_fp4_moe_weights(
            *weights, is_gated=False, intermediate_alignment=128, hidden_alignment=512
        )

        self.assertEqual(out[4:], (1920, 3072))
        for original, returned in zip(weights, out[:4]):
            self.assertIs(returned, original)

    def test_nemotron_pads_intermediate_and_hidden_with_zero_fill(self):
        num_experts, hidden, inter = 4, 2688, 1856
        padded_hidden, padded_inter = 3072, 1920
        w13, w13_scale, w2, w2_scale = _fp4_moe_weights(
            num_experts, hidden, inter, is_gated=False
        )

        p_w13, p_w13_scale, p_w2, p_w2_scale, out_inter, out_hidden = (
            _align_fp4_moe_weights(
                w13,
                w13_scale,
                w2,
                w2_scale,
                is_gated=False,
                intermediate_alignment=128,
                hidden_alignment=512,
            )
        )

        self.assertEqual((out_inter, out_hidden), (padded_inter, padded_hidden))
        # GEMM1: rows are intermediate, K is hidden (2 values/byte, 16/scale).
        self.assertEqual(p_w13.shape, (num_experts, padded_inter, padded_hidden // 2))
        self.assertEqual(
            p_w13_scale.shape, (num_experts, padded_inter, padded_hidden // 16)
        )
        torch.testing.assert_close(p_w13[:, :inter, : hidden // 2], w13)
        self.assertEqual(p_w13.sum().item(), w13.sum().item())
        self.assertEqual(
            p_w13_scale.view(torch.uint8).sum().item(),
            w13_scale.view(torch.uint8).sum().item(),
        )
        # GEMM2: rows are hidden, K is intermediate; padded rows/cols stay zero
        # so the sliced-off outputs never mix into real ones.
        self.assertEqual(p_w2.shape, (num_experts, padded_hidden, padded_inter // 2))
        self.assertEqual(
            p_w2_scale.shape, (num_experts, padded_hidden, padded_inter // 16)
        )
        torch.testing.assert_close(p_w2[:, :hidden, : inter // 2], w2)
        self.assertEqual(p_w2.sum().item(), w2.sum().item())
        self.assertEqual(
            p_w2_scale.view(torch.uint8).sum().item(),
            w2_scale.view(torch.uint8).sum().item(),
        )
        self.assertEqual(p_w13_scale.dtype, w13_scale.dtype)
        self.assertEqual(p_w2_scale.dtype, w2_scale.dtype)

    def test_gated_pads_only_intermediate_by_default(self):
        num_experts, hidden, inter = 2, 2688, 1000
        w13, w13_scale, w2, w2_scale = _fp4_moe_weights(
            num_experts, hidden, inter, is_gated=True
        )

        p_w13, _, p_w2, _, out_inter, out_hidden = _align_fp4_moe_weights(
            w13, w13_scale, w2, w2_scale, is_gated=True, intermediate_alignment=16
        )

        self.assertEqual((out_inter, out_hidden), (1008, hidden))
        self.assertEqual(p_w13.shape, (num_experts, 2 * 1008, hidden // 2))
        self.assertEqual(p_w2.shape, (num_experts, hidden, 1008 // 2))


if __name__ == "__main__":
    unittest.main()
