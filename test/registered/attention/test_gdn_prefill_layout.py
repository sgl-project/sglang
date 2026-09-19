import unittest
from unittest import mock

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

    def test_view_path_matches_contiguous_unpack(self):
        # The strided views must be indistinguishable from the fused unpack
        # copy they replace, for both consumers they feed on HIP: the QKV
        # split and the B/A gating.
        (qkvz, ba), (mixed_qkv, _, b, a) = self._projection_views(torch.bfloat16)
        split_args = (
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
        for view, ref in zip(
            fused_qkv_split_gdn_prefill(mixed_qkv, *split_args),
            fused_qkv_split_gdn_prefill(mixed_ref, *split_args),
        ):
            torch.testing.assert_close(view, ref, rtol=0, atol=0)

        a_log = torch.randn(self.NUM_V_HEADS, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(self.NUM_V_HEADS, dtype=torch.float32, device="cuda")
        for view, ref in zip(
            fused_gdn_gating(a_log, a, b, dt_bias),
            fused_gdn_gating(a_log, a.contiguous(), b.contiguous(), dt_bias),
        ):
            torch.testing.assert_close(view, ref, rtol=0, atol=0)

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
                torch.testing.assert_close(q, l2norm_fwd(q_ref), rtol=2e-2, atol=2e-3)
                torch.testing.assert_close(k, l2norm_fwd(k_ref), rtol=2e-2, atol=2e-3)
                for normalized in (q, k):
                    norms = normalized.float().pow(2).sum(-1).sqrt()
                    torch.testing.assert_close(
                        norms, torch.ones_like(norms), rtol=0, atol=5e-3
                    )

    def test_fused_split_l2norm_post_conv_layout(self):
        # forward_extend passes the post-conv tensor, a [T, qkv_dim] view of a
        # [qkv_dim, T] allocation, so the head dim is the strided axis rather
        # than the contiguous one. num_qk is cdiv(num_k_heads, attn_tp_size)
        # and need not be a power of two; 8/32 is the Qwen3.5 TP2 shape.
        for num_qk in (8, 6):
            for tokens in (0, 17):
                with self.subTest(num_qk=num_qk, tokens=tokens):
                    num_v, head = 4 * num_qk, self.HEAD_DIM
                    qkv_dim = 2 * num_qk * head + num_v * head
                    mixed_qkv = torch.randn(
                        qkv_dim, tokens, dtype=torch.bfloat16, device="cuda"
                    ).transpose(0, 1)

                    q, k, v = fused_qkv_split_l2norm_gdn_prefill(
                        mixed_qkv, num_qk, num_v, head, head
                    )
                    self.assertEqual(q.shape, (1, tokens, num_qk, head))
                    self.assertEqual(k.shape, (1, tokens, num_qk, head))
                    self.assertEqual(v.shape, (1, tokens, num_v, head))
                    if tokens == 0:
                        continue

                    self.assertEqual(mixed_qkv.stride(), (1, tokens))
                    q_ref, k_ref, v_ref = fused_qkv_split_gdn_prefill(
                        mixed_qkv, num_qk, num_qk, num_v, head, head, head
                    )
                    torch.testing.assert_close(v, v_ref, rtol=0, atol=0)
                    torch.testing.assert_close(
                        q, l2norm_fwd(q_ref), rtol=2e-2, atol=2e-3
                    )
                    torch.testing.assert_close(
                        k, l2norm_fwd(k_ref), rtol=2e-2, atol=2e-3
                    )


class TestGdnQkL2NormContract(unittest.TestCase):
    """extend() must default the Q/K norm on and forward it untouched.

    Re-hardcoding it drops the HIP fused-split saving silently, and losing
    the forward feeds unnormalized Q/K into the recurrence. No tensor-level
    test above can see either.
    """

    def _forwarded_norm_switch(self, **kwargs) -> bool:
        # Imported here so the contract check does not need a GPU build.
        from sglang.srt.layers.attention.linear.kernels import gdn_triton

        captured = {}

        def fake_chunk_gated_delta_rule(**call_kwargs):
            captured.update(call_kwargs)
            return None, None

        with mock.patch.object(
            gdn_triton, "chunk_gated_delta_rule", fake_chunk_gated_delta_rule
        ):
            gdn_triton.TritonGDNKernel().extend(
                q=None,
                k=None,
                v=None,
                g=None,
                beta=None,
                ssm_states=None,
                cache_indices=None,
                query_start_loc=None,
                **kwargs,
            )
        return captured["use_qk_l2norm_in_kernel"]

    def test_norm_switch_defaults_on_and_forwards(self):
        # Callers that did not pre-normalize keep the in-kernel norm.
        self.assertTrue(self._forwarded_norm_switch())
        # The HIP fused split pre-normalizes, so it switches the norm off.
        self.assertFalse(self._forwarded_norm_switch(use_qk_l2norm_in_kernel=False))


if __name__ == "__main__":
    unittest.main()
