"""MI35X correctness for GLM-5.3-Flash KDA W8A16 decode projections."""

import unittest

import torch

from sglang.srt.layers.quantization.unquant import (
    Glm53KdaPackedPtpcLinearMethod,
    Glm53KdaPtpcLinearMethod,
)
from sglang.srt.models.glm5_next import GLM53_KDA_W8A16_DECODE_CONFIGS
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(
    est_time=240,
    stage="stage-b",
    runner_config="1-gpu-small-amd-mi35x",
)

# Packed qkv/f_a/g_a layouts per local (N, K): (qkv, beta, fg) row counts.
PACKED_LAYOUTS = {(6416, 4096): (6144, 16, 256), (3344, 4096): (3072, 8, 256)}


def _token_counts(rows):
    counts, low = set(), 1
    for max_m, _ in rows:
        counts.update({low, (low + max_m) // 2, max_m})
        low = max_m + 1
    return sorted(counts)


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "requires one gfx950 GPU",
)
class TestGLM53KDAW8A16Decode(CustomTestCase):
    def _layer(self, n, k, seed):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        packed = PACKED_LAYOUTS.get((n, k))
        rows = sum(packed) if packed else n
        weight = (
            torch.randn(
                rows, k, generator=generator, device="cuda", dtype=torch.bfloat16
            )
            * 0.01
        )
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        table = GLM53_KDA_W8A16_DECODE_CONFIGS[(n, k)]
        if packed:
            qkv, beta, fg = packed
            method = Glm53KdaPackedPtpcLinearMethod(
                bf16_max_m=4095,
                fp8_max_m=16384,
                qkv_size=qkv,
                beta_size=beta,
                fg_size=fg,
                w8a16_configs=table,
            )
        else:
            method = Glm53KdaPtpcLinearMethod(
                "o_proj", bf16_max_m=5631, w8a16_configs=table
            )
        layer.quant_method = method
        method.process_weights_after_loading(layer)
        return layer, method, packed, generator

    @staticmethod
    def _dequantized(weight):
        import aiter

        fp8, scale = aiter.pertoken_quant(weight, quant_dtype=aiter.dtypes.fp8)
        return fp8.float() * scale

    def _decode(self, layer, method, packed, x, config):
        if packed:
            qkv, beta, fg = method.apply_w8a16_decode(layer, x, config)
            return torch.cat((qkv, beta, fg), dim=-1)
        return method.apply(layer, x)

    def test_matches_dequantized_reference_at_every_row_boundary(self):
        self.assertTrue(GLM53_KDA_W8A16_DECODE_CONFIGS)
        for (n, k), rows in GLM53_KDA_W8A16_DECODE_CONFIGS.items():
            layer, method, packed, generator = self._layer(n, k, seed=n + k)
            weight = layer.weight
            reference_weight = self._dequantized(weight)
            for m in _token_counts(rows):
                with self.subTest(n=n, k=k, m=m):
                    config = method.w8a16_config(m)
                    self.assertIsNotNone(config)
                    x = (
                        torch.randn(
                            m,
                            k,
                            generator=generator,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        * 0.1
                    )
                    actual = self._decode(layer, method, packed, x, config).float()
                    expected = x.float() @ reference_weight.t()
                    rel = (actual - expected).norm() / expected.norm()
                    self.assertLess(rel.item(), 5e-3)
                    bf16 = x.float() @ weight.float().t()
                    cosine = torch.nn.functional.cosine_similarity(
                        actual.flatten(), bf16.flatten(), dim=0
                    )
                    self.assertGreater(cosine.item(), 0.999)
                    self.assertTrue(torch.isfinite(actual).all())
            last_max_m = rows[-1][0]
            self.assertIsNone(method.w8a16_config(last_max_m + 1))
            self.assertIsNone(method.w8a16_config(0))

    def test_cuda_graph_replay_matches_eager(self):
        for (n, k), rows in GLM53_KDA_W8A16_DECODE_CONFIGS.items():
            layer, method, packed, generator = self._layer(n, k, seed=7 + n)
            for m in (1, rows[-1][0]):
                with self.subTest(n=n, k=k, m=m):
                    config = method.w8a16_config(m)
                    x = torch.randn(
                        m, k, generator=generator, device="cuda", dtype=torch.bfloat16
                    )
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        self._decode(layer, method, packed, x, config)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = self._decode(layer, method, packed, x, config)
                    x.copy_(
                        torch.randn(
                            m,
                            k,
                            generator=generator,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                    )
                    graph.replay()
                    torch.cuda.synchronize()
                    eager = self._decode(layer, method, packed, x, config)
                    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
