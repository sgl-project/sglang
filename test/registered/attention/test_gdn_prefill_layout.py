import os
import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention.fla import chunk_delta_h
from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.fla.l2norm import (
    gdn_prefill_qkv_prepare_fwd,
    l2norm_fwd,
)
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_gdn_prefill_prepare,
    fused_qkv_split_gdn_prefill,
    qwen3_5_gdn_prefill_projection_views,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=6, stage="base-b", runner_config="1-gpu-large")
# backend-specific: exercises ROCm strided views, split/gating fusion, and
# AMD chunk-state launch selection.
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGdnPrefillLayout(CustomTestCase):
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

    def test_fused_prepare_matches_reference(self):
        # Production passes a [T, qkv_dim] transpose of [qkv_dim, T].
        # Cases cover Qwen3.5 TP2, fp16, non-power-of-two heads, and empty input.
        cases = (
            (torch.bfloat16, 8, 17),
            (torch.float16, 2, 33),
            (torch.bfloat16, 6, 17),
            (torch.bfloat16, 8, 0),
        )
        for dtype, num_qk, tokens in cases:
            with self.subTest(dtype=dtype, num_qk=num_qk, tokens=tokens):
                num_v, head = 4 * num_qk, self.HEAD_DIM
                qkv_dim = (2 * num_qk + num_v) * head
                mixed_qkv = torch.randn(qkv_dim, tokens, dtype=dtype, device="cuda").T
                b, a = torch.randn(tokens, 2 * num_v, dtype=dtype, device="cuda").chunk(
                    2, dim=1
                )
                a_log = torch.randn(num_v, dtype=torch.float32, device="cuda")
                dt_bias = torch.zeros_like(a_log)
                if tokens:
                    a[0, :2] = a.new_tensor((100.0, 1.0))

                actual = fused_gdn_prefill_prepare(
                    mixed_qkv, a_log, a, b, dt_bias, num_qk, num_v, head, head
                )
                q, k, v, g, beta = actual
                self.assertEqual(
                    [x.shape for x in actual],
                    [(1, tokens, num_qk, head)] * 2
                    + [(1, tokens, num_v, head)]
                    + [(1, tokens, num_v)] * 2,
                )
                if not tokens:
                    continue

                self.assertEqual(mixed_qkv.stride(), (1, tokens))
                q_ref, k_ref, v_ref = fused_qkv_split_gdn_prefill(
                    mixed_qkv, num_qk, num_qk, num_v, head, head, head
                )
                g_ref, beta_ref = fused_gdn_gating(a_log, a, b, dt_bias)
                for out, ref in ((v, v_ref), (g, g_ref), (beta, beta_ref)):
                    torch.testing.assert_close(out, ref, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(g[0, 0, 0]))
                for out, ref in ((q, q_ref), (k, k_ref)):
                    torch.testing.assert_close(
                        out, l2norm_fwd(ref), rtol=2e-2, atol=2e-3
                    )


class TestGdnContracts(CustomTestCase):
    """Protect launch geometry and the split/chunk normalization contract."""

    def test_amd_grid_and_sequence_length_selectors(self):
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(chunk_delta_h, "_num_compute_units", return_value=256),
        ):
            os.environ.pop("SGLANG_GDN_CHUNK_H_BV", None)
            os.environ.pop("SGLANG_GDN_CHUNK_H_NUM_STAGES", None)
            select = chunk_delta_h._select_chunk_h_config
            self.assertEqual(
                [
                    select(*args)
                    for args in (
                        (1, 32, 8, 32, False, 0),
                        (1, 32, 12, 32, False, 0),
                        (1, 32, 128, 31, False, 0),
                        (2, 32, 128, 32, False, 0),
                        (4, 32, 128, 124, True, 0),
                        (4, 32, 128, 128, True, 0),
                    )
                ],
                [(16, 3), (16, 3), (16, 2), (32, 3), (64, 2), (64, 3)],
            )

    def _forwarded_norm_switch(self, **kwargs) -> bool:
        from sglang.srt.layers.attention.linear.kernels import gdn_triton

        with mock.patch.object(
            gdn_triton,
            "chunk_gated_delta_rule",
            side_effect=lambda **call: (call["use_qk_l2norm_in_kernel"], None),
        ):
            return gdn_triton.TritonGDNKernel().extend(
                *([None] * 5),
                ssm_states=None,
                cache_indices=None,
                query_start_loc=None,
                **kwargs,
            )[0]

    def test_norm_switch_defaults_on_and_forwards(self):
        self.assertTrue(self._forwarded_norm_switch())
        self.assertFalse(self._forwarded_norm_switch(use_qk_l2norm_in_kernel=False))


if __name__ == "__main__":
    unittest.main()
