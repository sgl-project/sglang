"""gfx950 correctness for the split-K-reduced BF16 x FP8 preshuffled GEMM."""

import unittest

import torch

from sglang.kernels.ops.gemm.gfx95_a16w8_preshuffle import (
    a16w8_preshuffle_gemm,
    a16w8_preshuffle_num_tiles,
    a16w8_preshuffle_supported,
)
from sglang.srt.models.glm5_next import GLM53_KDA_W8A16_DECODE_CONFIGS, _w8a16_config
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(
    est_time=180,
    stage="stage-b",
    runner_config="1-gpu-small-amd-mi35x",
)


def _cases():
    for (n, k), rows in GLM53_KDA_W8A16_DECODE_CONFIGS.items():
        low = 1
        for max_m, config in rows:
            for m in sorted({low, max_m}):
                yield n, k, m, config
            low = max_m + 1
    yield 6416, 4096, 8, _w8a16_config(8, 16, 1, 1, 3, 0)


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "requires one gfx950 GPU",
)
class TestGfx95A16W8Preshuffle(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import aiter
        from aiter.ops.shuffle import shuffle_weight

        cls.weights = {}
        for n, k in GLM53_KDA_W8A16_DECODE_CONFIGS:
            g = torch.Generator(device="cuda").manual_seed(n + k)
            weight = (
                torch.randn(n, k, generator=g, device="cuda", dtype=torch.bfloat16)
                * 0.01
            )
            fp8, scale = aiter.pertoken_quant(weight, quant_dtype=aiter.dtypes.fp8)
            cls.weights[(n, k)] = (
                shuffle_weight(fp8, (16, 16)).contiguous(),
                scale.contiguous(),
                fp8.float() * scale,
            )

    def test_split_k_reduction_matches_reference_and_repeats_bitwise(self):
        for n, k, m, config in _cases():
            with self.subTest(n=n, k=k, m=m, num_ksplit=config["NUM_KSPLIT"]):
                self.assertTrue(a16w8_preshuffle_supported(n, k, config))
                weight, scale, reference_weight = self.weights[(n, k)]
                counters = torch.zeros(
                    a16w8_preshuffle_num_tiles(m, n, config),
                    dtype=torch.int32,
                    device="cuda",
                )
                g = torch.Generator(device="cuda").manual_seed(m)
                x = (
                    torch.randn(m, k, generator=g, device="cuda", dtype=torch.bfloat16)
                    * 0.1
                )
                first = a16w8_preshuffle_gemm(x, weight, scale, config, counters)
                expected = x.float() @ reference_weight.t()
                rel = (first.float() - expected).norm() / expected.norm()
                self.assertLess(rel.item(), 5e-3)
                for _ in range(50):
                    again = a16w8_preshuffle_gemm(x, weight, scale, config, counters)
                    torch.testing.assert_close(again, first, rtol=0, atol=0)
                torch.cuda.synchronize()
                self.assertEqual(counters.abs().sum().item(), 0)

    def test_cuda_graph_replay_matches_eager(self):
        n, k = 6416, 4096
        weight, scale, _ = self.weights[(n, k)]
        for max_m, config in GLM53_KDA_W8A16_DECODE_CONFIGS[(n, k)]:
            with self.subTest(m=max_m):
                counters = torch.zeros(
                    a16w8_preshuffle_num_tiles(max_m, n, config),
                    dtype=torch.int32,
                    device="cuda",
                )
                x = torch.randn(max_m, k, device="cuda", dtype=torch.bfloat16)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    a16w8_preshuffle_gemm(x, weight, scale, config, counters)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = a16w8_preshuffle_gemm(x, weight, scale, config, counters)
                for _ in range(3):
                    x.copy_(torch.randn_like(x))
                    graph.replay()
                    torch.cuda.synchronize()
                    eager = a16w8_preshuffle_gemm(x, weight, scale, config, counters)
                    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
                self.assertEqual(counters.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
