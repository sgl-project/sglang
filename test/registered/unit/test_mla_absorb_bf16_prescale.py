"""Unit tests for the BF16 MLA-absorb prescale-fold + flatten-view optimization.

These validate the two transformations the optimization relies on, both of which
must be *bit-identical* to the original per-decode-step form:

1. Load-time scale fold: precomputing ``w.to(bfloat16) * w_scale`` once and
   reusing it produces the same absorb-BMM output as recomputing it every step.
2. V-absorb flatten-view: writing the BMM into a ``(batch, heads, vdim)`` buffer
   through a transposed view and flattening it is bit-identical to
   ``bmm(...).transpose(0, 1).flatten(1, 2)`` and avoids the extra copy (the
   flatten is a view, not a materialized copy).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


class TestMlaAbsorbBf16Prescale(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        # (num_heads, batch, kv_lora, qk_nope_dim, v_dim) — small but shaped like
        # the DeepSeek MLA absorb BMMs (batched over heads).
        self.H, self.M, self.L, self.D, self.Dv = 16, 32, 24, 20, 18

    def test_prescale_fold_is_bitwise_identical(self):
        """Folding the constant scale into w_kc once == per-step dequant."""
        H, M, L, D = self.H, self.M, self.L, self.D
        q_nope = torch.randn(M, H, L, dtype=torch.bfloat16)
        w_kc = torch.randn(H, L, D, dtype=torch.bfloat16)

        for w_scale in (torch.tensor(1.0), torch.tensor(1.7), torch.tensor(2.0)):
            x = q_nope.to(torch.bfloat16).transpose(0, 1)  # (H, M, L)

            # Original: recompute dequant every "step".
            ref = torch.bmm(x, w_kc.to(torch.bfloat16) * w_scale)

            # Optimized: fold once at load, forward consumes it directly.
            w_kc_folded = w_kc.to(torch.bfloat16) * w_scale
            got = torch.bmm(x, w_kc_folded)

            self.assertTrue(
                torch.equal(ref, got),
                msg=f"prescale fold mismatch at w_scale={float(w_scale)}",
            )

    def test_vabsorb_flatten_view_is_bitwise_identical_and_copy_free(self):
        """out=buf.transpose(...) + buf.flatten == transpose(0,1).flatten(1,2)."""
        H, M, L, Dv = self.H, self.M, self.L, self.Dv
        attn_output = torch.randn(M, H, L, dtype=torch.bfloat16)
        w_vc = torch.randn(H, L, Dv, dtype=torch.bfloat16)
        w_scale = torch.tensor(2.0)

        a = attn_output.to(torch.bfloat16).transpose(0, 1)  # (H, M, L)
        w_vc_folded = w_vc.to(torch.bfloat16) * w_scale

        # Original: BMM then transpose+flatten (materializes a copy).
        ref = torch.bmm(a, w_vc_folded).transpose(0, 1).flatten(1, 2)  # (M, H*Dv)

        # Optimized: BMM straight into a (batch, heads, vdim) buffer via a
        # transposed view, then flatten as a free view.
        buf = torch.empty(M, H, Dv, dtype=torch.bfloat16)
        torch.bmm(a, w_vc_folded, out=buf.transpose(0, 1))
        got = buf.flatten(1, 2)  # (M, H*Dv)

        self.assertEqual(tuple(got.shape), (M, H * Dv))
        self.assertTrue(torch.equal(ref, got), msg="flatten-view result mismatch")

        # The optimized flatten must be a view over buf's storage (no copy).
        self.assertEqual(
            got.untyped_storage().data_ptr(),
            buf.untyped_storage().data_ptr(),
            msg="flatten(1, 2) unexpectedly materialized a copy",
        )

    def test_end_to_end_absorb_equivalence(self):
        """Both stages combined match the unoptimized reference bit-for-bit."""
        H, M, L, D, Dv = self.H, self.M, self.L, self.D, self.Dv
        q_nope = torch.randn(M, H, L, dtype=torch.bfloat16)
        attn_output = torch.randn(M, H, L, dtype=torch.bfloat16)
        w_kc = torch.randn(H, L, D, dtype=torch.bfloat16)
        w_vc = torch.randn(H, L, Dv, dtype=torch.bfloat16)
        w_scale = torch.tensor(1.7)

        # Reference (per-step dequant + transpose/flatten copy).
        qx = q_nope.to(torch.bfloat16).transpose(0, 1)
        ax = attn_output.to(torch.bfloat16).transpose(0, 1)
        ref_q = torch.bmm(qx, w_kc.to(torch.bfloat16) * w_scale)
        ref_v = torch.bmm(ax, w_vc.to(torch.bfloat16) * w_scale)
        ref_v = ref_v.transpose(0, 1).flatten(1, 2)

        # Optimized (folded weights + flatten-view).
        w_kc_f = w_kc.to(torch.bfloat16) * w_scale
        w_vc_f = w_vc.to(torch.bfloat16) * w_scale
        got_q = torch.bmm(qx, w_kc_f)
        buf = torch.empty(M, H, Dv, dtype=torch.bfloat16)
        torch.bmm(ax, w_vc_f, out=buf.transpose(0, 1))
        got_v = buf.flatten(1, 2)

        self.assertTrue(torch.equal(ref_q, got_q), msg="q_nope absorb mismatch")
        self.assertTrue(torch.equal(ref_v, got_v), msg="v absorb mismatch")


if __name__ == "__main__":
    unittest.main()
