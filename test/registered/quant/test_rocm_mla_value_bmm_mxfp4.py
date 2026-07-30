import unittest
from unittest import mock

import torch

from sglang.srt.utils.common import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestRocmMlaValueBmmMxfp4(CustomTestCase):
    def test_fused_bmm_quant_matches_split_path(self):
        if not is_gfx95_supported():
            self.skipTest("requires gfx95 MXFP4 hardware")

        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        from sglang.srt.layers.quantization import rocm_mla_value_bmm_mxfp4
        from sglang.srt.layers.quantization.rocm_mla_value_bmm_mxfp4 import (
            batched_gemm_a16wfp4_flatten_mxfp4_quant,
            can_fuse_mla_value_bmm_mxfp4_quant,
        )
        from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
            batched_gemm_afp4wfp4_pre_quant,
            fused_flatten_mxfp4_quant,
        )

        torch.manual_seed(7)
        heads, n, k = 4, 128, 512
        weight = torch.randn((heads * n, k), dtype=torch.bfloat16, device="cuda")
        weight_fp4, weight_scales = dynamic_mxfp4_quant(weight)
        weight_fp4 = weight_fp4.view(heads, n, k // 2)
        weight_scales = weight_scales.view(heads, n, k // 32)

        for tokens in (1, 17, 64):
            with self.subTest(tokens=tokens):
                x = torch.randn((heads, tokens, k), dtype=torch.bfloat16, device="cuda")
                self.assertTrue(
                    can_fuse_mla_value_bmm_mxfp4_quant(x, weight_fp4, weight_scales)
                )

                bf16_output = torch.empty(
                    (tokens, heads, n), dtype=torch.bfloat16, device="cuda"
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    weight_fp4,
                    weight_scales,
                    torch.bfloat16,
                    bf16_output.transpose(0, 1),
                )
                expected_fp4, expected_scales = fused_flatten_mxfp4_quant(bf16_output)

                actual_fp4, actual_scales = batched_gemm_a16wfp4_flatten_mxfp4_quant(
                    x, weight_fp4, weight_scales
                )
                torch.testing.assert_close(actual_fp4, expected_fp4, rtol=0, atol=0)
                torch.testing.assert_close(
                    actual_scales, expected_scales, rtol=0, atol=0
                )
                self.assertEqual(actual_scales.stride(), expected_scales.stride())

        x = torch.randn((heads, 8, k), dtype=torch.bfloat16, device="cuda")
        compiled = torch.compile(
            batched_gemm_a16wfp4_flatten_mxfp4_quant, fullgraph=True
        )
        compiled_fp4, compiled_scales = compiled(x, weight_fp4, weight_scales)

        bf16_output = torch.empty((8, heads, n), dtype=torch.bfloat16, device="cuda")
        batched_gemm_afp4wfp4_pre_quant(
            x,
            weight_fp4,
            weight_scales,
            torch.bfloat16,
            bf16_output.transpose(0, 1),
        )
        expected_fp4, expected_scales = fused_flatten_mxfp4_quant(bf16_output)
        torch.testing.assert_close(compiled_fp4, expected_fp4, rtol=0, atol=0)
        torch.testing.assert_close(compiled_scales, expected_scales, rtol=0, atol=0)

        with mock.patch.object(
            rocm_mla_value_bmm_mxfp4, "_get_fused_config", return_value=None
        ):
            fallback_fp4, fallback_scales = batched_gemm_a16wfp4_flatten_mxfp4_quant(
                x, weight_fp4, weight_scales
            )
        torch.testing.assert_close(fallback_fp4, expected_fp4, rtol=0, atol=0)
        torch.testing.assert_close(fallback_scales, expected_scales, rtol=0, atol=0)

        # Warm the Triton/custom-op path before graph capture.
        batched_gemm_a16wfp4_flatten_mxfp4_quant(x, weight_fp4, weight_scales)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_fp4, graph_scales = batched_gemm_a16wfp4_flatten_mxfp4_quant(
                x, weight_fp4, weight_scales
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_fp4, expected_fp4, rtol=0, atol=0)
        torch.testing.assert_close(graph_scales, expected_scales, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
