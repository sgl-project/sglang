import unittest

import torch

from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.fla.l2norm import (
    gdn_prefill_qkv_prepare_fwd,
    l2norm_fwd,
)
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
    fused_qkv_split_l2norm_gdn_prefill,
    fused_qkvzba_split_reshape_cat_contiguous,
    qwen3_5_gdn_prefill_projection_views,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGdnPrefillLayout(unittest.TestCase):
    TOKENS = 33
    NUM_QK_HEADS = 2
    NUM_V_HEADS = 4
    HEAD_DIM = 128

    def _projection_views(self, dtype):
        qkv_dim = (
            2 * self.NUM_QK_HEADS * self.HEAD_DIM + self.NUM_V_HEADS * self.HEAD_DIM
        )
        qkvz = torch.randn(
            self.TOKENS,
            qkv_dim + self.NUM_V_HEADS * self.HEAD_DIM,
            dtype=dtype,
            device="cuda",
        )
        ba = torch.randn(
            self.TOKENS,
            2 * self.NUM_V_HEADS,
            dtype=dtype,
            device="cuda",
        )
        return (qkvz, ba), qwen3_5_gdn_prefill_projection_views(
            qkvz,
            ba,
            self.NUM_QK_HEADS,
            self.NUM_V_HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
        )

    def test_projection_views_preserve_layout(self):
        (qkvz, ba), (mixed_qkv, z, b, a) = self._projection_views(torch.bfloat16)
        qkv_dim = mixed_qkv.shape[1]

        self.assertFalse(mixed_qkv.is_contiguous())
        self.assertFalse(z.is_contiguous())
        self.assertFalse(b.is_contiguous())
        self.assertFalse(a.is_contiguous())
        torch.testing.assert_close(mixed_qkv, qkvz[:, :qkv_dim], rtol=0, atol=0)
        torch.testing.assert_close(
            z.reshape(self.TOKENS, -1), qkvz[:, qkv_dim:], rtol=0, atol=0
        )
        torch.testing.assert_close(b, ba[:, : self.NUM_V_HEADS], rtol=0, atol=0)
        torch.testing.assert_close(a, ba[:, self.NUM_V_HEADS :], rtol=0, atol=0)

    def test_qkv_prepare_preserves_dtype_and_matches_materialized_path(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                _, (mixed_qkv, _, _, _) = self._projection_views(dtype)
                q_dim = self.NUM_QK_HEADS * self.HEAD_DIM
                v_dim = self.NUM_V_HEADS * self.HEAD_DIM
                q = mixed_qkv[:, :q_dim].view(
                    self.TOKENS, self.NUM_QK_HEADS, self.HEAD_DIM
                )
                k = mixed_qkv[:, q_dim : 2 * q_dim].view(
                    self.TOKENS, self.NUM_QK_HEADS, self.HEAD_DIM
                )
                v = mixed_qkv[:, 2 * q_dim : 2 * q_dim + v_dim].view(
                    self.TOKENS, self.NUM_V_HEADS, self.HEAD_DIM
                )

                q_out, k_out, v_out = gdn_prefill_qkv_prepare_fwd(q, k, v)

                self.assertEqual(q_out.dtype, dtype)
                self.assertEqual(k_out.dtype, dtype)
                self.assertEqual(v_out.dtype, dtype)
                torch.testing.assert_close(
                    q_out, l2norm_fwd(q.contiguous()), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    k_out, l2norm_fwd(k.contiguous()), rtol=0, atol=0
                )
                torch.testing.assert_close(v_out, v.contiguous(), rtol=0, atol=0)

    def test_fused_split_flashinfer_prepare_reuses_contiguous_value(self):
        _, (mixed_qkv, _, _, _) = self._projection_views(torch.bfloat16)
        q, k, v = fused_qkv_split_gdn_prefill(
            mixed_qkv,
            self.NUM_QK_HEADS,
            self.NUM_QK_HEADS,
            self.NUM_V_HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            self.HEAD_DIM,
        )

        q_out, k_out, v_out = gdn_prefill_qkv_prepare_fwd(q[0], k[0], v[0])

        self.assertEqual(v_out.data_ptr(), v.data_ptr())
        torch.testing.assert_close(q_out, l2norm_fwd(q[0]), rtol=0, atol=0)
        torch.testing.assert_close(k_out, l2norm_fwd(k[0]), rtol=0, atol=0)

    def test_strided_gate_matches_contiguous_gate(self):
        for dtype in (torch.bfloat16, torch.float16):
            for norm_before_gate in (True, False):
                with self.subTest(dtype=dtype, norm_before_gate=norm_before_gate):
                    _, (_, z, _, _) = self._projection_views(dtype)
                    x = torch.randn(
                        self.TOKENS * self.NUM_V_HEADS,
                        self.HEAD_DIM,
                        dtype=dtype,
                        device="cuda",
                    )
                    weight = torch.randn(self.HEAD_DIM, dtype=dtype, device="cuda")
                    expected = rms_norm_gated(
                        x=x,
                        weight=weight,
                        bias=None,
                        z=z.contiguous().view_as(x),
                        norm_before_gate=norm_before_gate,
                        is_rms_norm=True,
                    )
                    actual = rms_norm_gated(
                        x=x,
                        weight=weight,
                        bias=None,
                        z=z,
                        norm_before_gate=norm_before_gate,
                        is_rms_norm=True,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_strided_gating_matches_contiguous(self):
        _, (_, _, b, a) = self._projection_views(torch.bfloat16)
        a_log = torch.randn(self.NUM_V_HEADS, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(self.NUM_V_HEADS, dtype=torch.float32, device="cuda")
        g_view, beta_view = fused_gdn_gating(a_log, a, b, dt_bias)
        g_ref, beta_ref = fused_gdn_gating(
            a_log, a.contiguous(), b.contiguous(), dt_bias
        )
        torch.testing.assert_close(g_view, g_ref, rtol=0, atol=0)
        torch.testing.assert_close(beta_view, beta_ref, rtol=0, atol=0)

    def test_fused_split_from_strided_mixed_qkv_matches_unpack(self):
        (qkvz, ba), (mixed_qkv, _, _, _) = self._projection_views(torch.bfloat16)
        q_view, k_view, v_view = fused_qkv_split_gdn_prefill(
            mixed_qkv,
            self.NUM_QK_HEADS,
            self.NUM_QK_HEADS,
            self.NUM_V_HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            self.HEAD_DIM,
        )
        mixed_ref, _, _, _ = fused_qkvzba_split_reshape_cat_contiguous(
            qkvz,
            ba,
            self.NUM_QK_HEADS,
            self.NUM_V_HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
        )
        q_ref, k_ref, v_ref = fused_qkv_split_gdn_prefill(
            mixed_ref,
            self.NUM_QK_HEADS,
            self.NUM_QK_HEADS,
            self.NUM_V_HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            self.HEAD_DIM,
        )
        torch.testing.assert_close(q_view, q_ref, rtol=0, atol=0)
        torch.testing.assert_close(k_view, k_ref, rtol=0, atol=0)
        torch.testing.assert_close(v_view, v_ref, rtol=0, atol=0)

    def test_fused_split_l2norm_matches_split_then_l2norm(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                _, (mixed_qkv, _, _, _) = self._projection_views(dtype)
                q_ref, k_ref, v_ref = fused_qkv_split_gdn_prefill(
                    mixed_qkv,
                    self.NUM_QK_HEADS,
                    self.NUM_QK_HEADS,
                    self.NUM_V_HEADS,
                    self.HEAD_DIM,
                    self.HEAD_DIM,
                    self.HEAD_DIM,
                )
                q, k, v = fused_qkv_split_l2norm_gdn_prefill(
                    mixed_qkv,
                    self.NUM_QK_HEADS,
                    self.NUM_V_HEADS,
                    self.HEAD_DIM,
                    self.HEAD_DIM,
                )

                self.assertEqual(q.dtype, dtype)
                self.assertEqual(k.dtype, dtype)
                torch.testing.assert_close(v, v_ref, rtol=0, atol=0)
                # Fusing the norm changes the reduction block shape, so Q/K
                # land within an ulp of the two-launch path rather than on it.
                torch.testing.assert_close(
                    q, l2norm_fwd(q_ref), rtol=2e-2, atol=2e-3
                )
                torch.testing.assert_close(
                    k, l2norm_fwd(k_ref), rtol=2e-2, atol=2e-3
                )
                for normalized in (q, k):
                    norms = normalized.float().pow(2).sum(-1).sqrt()
                    torch.testing.assert_close(
                        norms, torch.ones_like(norms), rtol=0, atol=5e-3
                    )

    def test_fused_split_l2norm_qwen35_tp2_shape_and_empty_batch(self):
        num_qk, num_v, head = 8, 32, 128
        qkv_dim = 2 * num_qk * head + num_v * head
        for tokens in (0, 17):
            with self.subTest(tokens=tokens):
                qkvz = torch.randn(
                    tokens,
                    qkv_dim + num_v * head,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                mixed_qkv = qkvz[:, :qkv_dim]
                q, k, v = fused_qkv_split_l2norm_gdn_prefill(
                    mixed_qkv, num_qk, num_v, head, head
                )
                self.assertEqual(q.shape, (1, tokens, num_qk, head))
                self.assertEqual(k.shape, (1, tokens, num_qk, head))
                self.assertEqual(v.shape, (1, tokens, num_v, head))
                if tokens == 0:
                    continue
                torch.testing.assert_close(
                    v[0].reshape(tokens, -1),
                    mixed_qkv[:, 2 * num_qk * head :],
                    rtol=0,
                    atol=0,
                )

    def test_qwen35_tp2_ratio4_views_and_empty_batch(self):
        num_qk, num_v, head = 8, 32, 128
        qkv_dim = 2 * num_qk * head + num_v * head
        for tokens in (0, 17):
            with self.subTest(tokens=tokens):
                qkvz = torch.randn(
                    tokens,
                    qkv_dim + num_v * head,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                ba = torch.randn(tokens, 2 * num_v, dtype=torch.bfloat16, device="cuda")
                mixed_qkv, z, b, a = qwen3_5_gdn_prefill_projection_views(
                    qkvz, ba, num_qk, num_v, head, head
                )
                self.assertEqual(mixed_qkv.shape, (tokens, qkv_dim))
                self.assertEqual(z.shape, (tokens, num_v, head))
                self.assertEqual(b.shape, (tokens, num_v))
                self.assertEqual(a.shape, (tokens, num_v))
                if tokens == 0:
                    continue
                self.assertFalse(mixed_qkv.is_contiguous())
                torch.testing.assert_close(mixed_qkv, qkvz[:, :qkv_dim], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
