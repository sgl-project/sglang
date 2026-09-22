"""MI355X correctness for the gated RMSNorm-to-PTPC bridge."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.fused_norm_gate import (
    FusedRMSNormGated,
    rms_norm_gated_per_token_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import apply_fp8_ptpc_linear
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="jit-kernel-unit-test-amd")


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "requires one gfx950 GPU",
)
class TestGLM53KDAOProjNormQuant(CustomTestCase):
    @staticmethod
    def _reference(x, gate, norm):
        import aiter

        normalized = norm(x.clone(), gate).flatten(1)
        return aiter.per_token_quant_hip(
            normalized,
            quant_dtype=aiter.dtypes.fp8,
        )

    def _check_bridge(self, m: int, heads: int, seed: int, extreme_gate: bool):
        torch.manual_seed(seed)
        x = torch.randn(m, heads, 128, dtype=torch.bfloat16, device="cuda") * 0.1
        gate = torch.randn_like(x)
        if extreme_gate and m:
            gate[::2].fill_(20)
            gate[1::2].fill_(-20)
        norm = FusedRMSNormGated(
            128,
            eps=1e-5,
            activation="sigmoid",
            device="cuda",
            dtype=torch.bfloat16,
        )
        norm.weight.data.copy_(torch.randn_like(norm.weight))

        expected = self._reference(x, gate, norm) if m else None
        actual_q, actual_scale = rms_norm_gated_per_token_fp8(
            x,
            gate,
            norm.weight,
            norm.eps,
        )
        self.assertEqual(actual_q.shape, (m, heads * 128))
        self.assertEqual(actual_scale.shape, (m, 1))
        self.assertEqual(actual_q.dtype, torch.float8_e4m3fn)
        self.assertEqual(actual_scale.dtype, torch.float32)
        if m:
            expected_q, expected_scale = expected
            actual = actual_q.float() * actual_scale
            expected = expected_q.float() * expected_scale
            cosine = torch.nn.functional.cosine_similarity(
                actual.flatten(), expected.flatten(), dim=0
            )
            mean_abs = (actual - expected).abs().mean()
            self.assertGreater(cosine.item(), 0.9998)
            self.assertLess(mean_abs.item(), 0.002)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue((actual_scale > 0).all())

    def test_tp4_tp8_layouts_and_boundaries(self):
        for heads in (8, 16):
            for m in (0, 1, 256, 8192):
                for seed in (0, 17):
                    with self.subTest(heads=heads, m=m, seed=seed):
                        self._check_bridge(
                            m=m,
                            heads=heads,
                            seed=seed,
                            extreme_gate=False,
                        )

    def test_extreme_gates(self):
        for heads in (8, 16):
            with self.subTest(heads=heads):
                self._check_bridge(
                    m=256,
                    heads=heads,
                    seed=29,
                    extreme_gate=True,
                )

    def test_long_context(self):
        for heads in (8, 16):
            with self.subTest(heads=heads):
                self._check_bridge(
                    m=131072,
                    heads=heads,
                    seed=41,
                    extreme_gate=False,
                )
                torch.cuda.empty_cache()

    def test_final_ptpc_output_and_cuda_graph(self):
        import aiter
        from aiter.ops.shuffle import shuffle_weight

        m, heads, output_size = 512, 8, 256
        torch.manual_seed(53)
        x = torch.randn(m, heads, 128, dtype=torch.bfloat16, device="cuda") * 0.1
        gate = torch.randn_like(x)
        norm = FusedRMSNormGated(
            128,
            eps=1e-5,
            activation="sigmoid",
            device="cuda",
            dtype=torch.bfloat16,
        )
        norm.weight.data.copy_(torch.randn_like(norm.weight))
        weight = (
            torch.randn(
                output_size,
                heads * 128,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.01
        )
        fp8_weight, weight_scale = aiter.pertoken_quant(
            weight,
            quant_dtype=aiter.dtypes.fp8,
        )
        fp8_weight = shuffle_weight(fp8_weight, (16, 16)).contiguous()

        expected_input = self._reference(x, gate, norm)
        expected = apply_fp8_ptpc_linear(
            expected_input,
            fp8_weight,
            weight_scale,
        )

        def run():
            return apply_fp8_ptpc_linear(
                rms_norm_gated_per_token_fp8(
                    x,
                    gate,
                    norm.weight,
                    norm.eps,
                ),
                fp8_weight,
                weight_scale,
            )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        graph.replay()
        torch.cuda.synchronize()

        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(),
            expected.float().flatten(),
            dim=0,
        )
        mean_abs = (actual.float() - expected.float()).abs().mean()
        self.assertGreater(cosine.item(), 0.999)
        self.assertLess(mean_abs.item(), 0.01)
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main(verbosity=3)
