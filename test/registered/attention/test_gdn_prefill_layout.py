import unittest

import torch

from sglang.kernels.ops.attention.fla.l2norm import (
    gdn_prefill_qkv_prepare_fwd,
    l2norm_fwd,
)
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
    qwen3_5_gdn_prefill_projection_views,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")


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


if __name__ == "__main__":
    unittest.main()
