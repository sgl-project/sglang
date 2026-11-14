"""The fused MLA value BMM + MXFP4 output quant is bit-identical to the split path."""

import unittest
from unittest import mock

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

HEADS, N, K = 4, 128, 512


@unittest.skipUnless(is_gfx95_supported(), "requires gfx95 MXFP4 hardware")
class TestMlaValueBmmMxfp4Hip(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        torch.manual_seed(7)
        weight = torch.randn((HEADS * N, K), dtype=torch.bfloat16, device="cuda")
        weight_fp4, weight_scales = dynamic_mxfp4_quant(weight)
        cls.weight_fp4 = weight_fp4.view(HEADS, N, K // 2)
        cls.weight_scales = weight_scales.view(HEADS, N, K // 32)

    def _split_path(self, x, w=None, w_scales=None):
        from aiter.ops.triton.batched_gemm_afp4wfp4_pre_quant import (
            batched_gemm_afp4wfp4_pre_quant,
        )
        from aiter.ops.triton.fused_mxfp4_quant import fused_flatten_mxfp4_quant

        bf16_output = torch.empty(
            (x.shape[1], HEADS, N), dtype=torch.bfloat16, device="cuda"
        )
        batched_gemm_afp4wfp4_pre_quant(
            x,
            self.weight_fp4 if w is None else w,
            self.weight_scales if w_scales is None else w_scales,
            torch.bfloat16,
            bf16_output.transpose(0, 1),
        )
        return fused_flatten_mxfp4_quant(bf16_output)

    def _fused(self, x, w=None, w_scales=None):
        from sglang.kernels.ops.gemm.mla_value_bmm_mxfp4_hip import (
            batched_gemm_a16wfp4_flatten_mxfp4_quant,
        )

        return batched_gemm_a16wfp4_flatten_mxfp4_quant(
            x,
            self.weight_fp4 if w is None else w,
            self.weight_scales if w_scales is None else w_scales,
        )

    def _assert_matches_split_path(
        self, x, actual_fp4, actual_scales, w=None, w_scales=None
    ):
        expected_fp4, expected_scales = self._split_path(x, w, w_scales)
        torch.testing.assert_close(actual_fp4, expected_fp4, rtol=0, atol=0)
        torch.testing.assert_close(actual_scales, expected_scales, rtol=0, atol=0)
        # o_proj consumes the transposed scale view without materializing it.
        self.assertEqual(actual_scales.stride(), expected_scales.stride())

    def _x(self, tokens):
        return torch.randn((HEADS, tokens, K), dtype=torch.bfloat16, device="cuda")

    def test_matches_split_path_per_token_count(self):
        from sglang.kernels.ops.gemm.mla_value_bmm_mxfp4_hip import (
            can_fuse_mla_value_bmm_mxfp4_quant,
        )

        # 17 leaves a partial M tile; 64 spans several.
        for tokens in (1, 17, 64):
            with self.subTest(tokens=tokens):
                x = self._x(tokens)
                self.assertTrue(
                    can_fuse_mla_value_bmm_mxfp4_quant(
                        x, self.weight_fp4, self.weight_scales
                    )
                )
                self._assert_matches_split_path(x, *self._fused(x))

    def test_matches_split_path_on_mla_call_site_views(self):
        """forward_mla_rocm passes strided views, not the contiguous tensors above."""
        from sglang.kernels.ops.gemm.mla_value_bmm_mxfp4_hip import (
            can_fuse_mla_value_bmm_mxfp4_quant,
        )

        # The weight loader stores w_vc / w_scale_v transposed and the call site
        # transposes them back; x is a head-major view of attn_output (tokens, heads, K).
        w_vc = self.weight_fp4.contiguous().transpose(1, 2)
        w_scale_v = self.weight_scales.contiguous().transpose(1, 2)
        w, w_scales = w_vc.transpose(-2, -1), w_scale_v.transpose(-2, -1)
        for tokens in (1, 17, 64):
            with self.subTest(tokens=tokens):
                attn_output = torch.randn(
                    (tokens, HEADS, K), dtype=torch.bfloat16, device="cuda"
                )
                x = attn_output.transpose(0, 1)
                self.assertEqual(x.stride(), (K, HEADS * K, 1))
                self.assertTrue(can_fuse_mla_value_bmm_mxfp4_quant(x, w, w_scales))
                self._assert_matches_split_path(
                    x, *self._fused(x, w, w_scales), w=w, w_scales=w_scales
                )

    def test_torch_compile_fullgraph(self):
        from sglang.kernels.ops.gemm.mla_value_bmm_mxfp4_hip import (
            batched_gemm_a16wfp4_flatten_mxfp4_quant,
        )

        x = self._x(8)
        compiled = torch.compile(
            batched_gemm_a16wfp4_flatten_mxfp4_quant, fullgraph=True
        )
        self._assert_matches_split_path(
            x, *compiled(x, self.weight_fp4, self.weight_scales)
        )

    def test_config_lookup_fallback_uses_split_path(self):
        from sglang.kernels.ops.gemm import mla_value_bmm_mxfp4_hip

        x = self._x(8)
        with mock.patch.object(
            mla_value_bmm_mxfp4_hip, "_get_fused_config", return_value=None
        ):
            fallback = self._fused(x)
        self._assert_matches_split_path(x, *fallback)

    def test_cuda_graph_replay(self):
        x = self._x(8)
        # Warm the Triton/custom-op path before graph capture.
        self._fused(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_fp4, graph_scales = self._fused(x)
        graph.replay()
        torch.cuda.synchronize()
        self._assert_matches_split_path(x, graph_fp4, graph_scales)


if __name__ == "__main__":
    unittest.main()
