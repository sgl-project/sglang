import os
import unittest
from unittest import mock

import torch

from sglang.kernels.ops.gemm import mimo_fused_qkv_gfx950 as mono
from sglang.srt.layers.quantization import fp8_utils
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

N = 3392
K = 6144
MS = (1, 2, 4, 8, 16)


def _input(m):
    magnitudes = torch.rand(m, K // 128, 1, device="cuda") * 8
    return (
        torch.randn(m, K // 128, 128, device="cuda") * magnitudes
    ).view(m, K).bfloat16()


def _weight():
    import aiter
    from aiter.ops.shuffle import shuffle_weight

    weight = (torch.randn(N, K, device="cuda") * 0.5).to(aiter.dtypes.fp8)
    weight = shuffle_weight(weight, (16, 16))
    weight.is_shuffled = True
    return weight


@unittest.skipUnless(
    torch.version.hip
    and fp8_utils._use_aiter_bpreshuffle_gfx95
    and torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    "gfx950 with AITER block-FP8 bpreshuffle only",
)
class TestMiMoMonoQKVGfx950(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_ROCM_MIMO_FUSED_QKV"] = "1"
        torch.manual_seed(0)
        cls.weight = _weight()
        cls.weight_scale = (
            torch.rand((N + 127) // 128, K // 128, device="cuda") * 1e-2 + 1e-3
        )

    def _reference(self, x):
        import aiter

        quantized, scale = fp8_utils.aiter_per1x128_quant(
            x, quant_dtype=aiter.dtypes.fp8, transpose_scale=False
        )
        scale = fp8_utils.materialize_bpreshuffle_fp8_scale(scale)
        return aiter.gemm_a8w8_blockscale_bpreshuffle(
            quantized,
            self.weight,
            scale,
            self.weight_scale,
            dtype=torch.bfloat16,
        )

    def test_matches_aiter(self):
        for m in MS:
            with self.subTest(m=m):
                x = _input(m)
                expected = self._reference(x)
                actual = mono.try_smallm_fp8_bs_linear(
                    x, self.weight, self.weight_scale
                )
                self.assertIsNotNone(actual)
                torch.testing.assert_close(
                    actual, expected, atol=2e-2, rtol=2e-2
                )

    def test_fallback_guards(self):
        with mock.patch.dict(os.environ, {"SGLANG_ROCM_MIMO_FUSED_QKV": "0"}):
            self.assertIsNone(
                mono.try_smallm_fp8_bs_linear(
                    _input(4), self.weight, self.weight_scale
                )
            )
        self.assertIsNone(
            mono.try_smallm_fp8_bs_linear(
                _input(17), self.weight, self.weight_scale
            )
        )

    def test_graph_replay_matches_eager(self):
        x = _input(4)
        eager = mono.try_smallm_fp8_bs_linear(
            x, self.weight, self.weight_scale
        )
        self.assertIsNotNone(eager)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mono.try_smallm_fp8_bs_linear(
                x, self.weight, self.weight_scale
            )
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(captured, eager))


if __name__ == "__main__":
    unittest.main()
