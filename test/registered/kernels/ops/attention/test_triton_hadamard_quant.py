"""Correctness tests for fused DSA query Hadamard and FP8 quantization."""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.triton_hadamard_quant import (
    _FP8_DTYPE,
    fused_hadamard_act_quant,
)
from sglang.kernels.ops.quantization.act_quant_tilelang import act_quant
from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: exercises the gfx950 FP8 FNUZ and TileLang reference path.
register_amd_ci(
    est_time=180,
    stage="stage-b",
    runner_config="1-gpu-small-amd-mi35x",
)

_BLOCK = 128
_HEADS = 32
_ROWS = (1, 64, 4096, 16384)
_SEEDS = (0, 41)


def _production_reference(x: torch.Tensor, scale_fmt):
    return act_quant(rotate_activation(x), _BLOCK, scale_fmt)


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "requires one gfx950 GPU",
)
class TestFusedHadamardActQuant(CustomTestCase):
    def _input(self, rows, seed):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return (
            torch.randn(
                rows,
                _HEADS,
                _BLOCK,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            * 0.3
        ).contiguous()

    def _assert_quantized_close(self, y, scale, y_ref, scale_ref):
        self.assertEqual(y.dtype, _FP8_DTYPE)
        self.assertTrue(torch.isfinite(y.float()).all().item())
        self.assertTrue(torch.isfinite(scale).all().item())

        dequant = y.float() * scale
        dequant_ref = y_ref.float() * scale_ref
        cosine = torch.nn.functional.cosine_similarity(
            dequant.flatten(), dequant_ref.flatten(), dim=0
        ).item()
        self.assertGreater(cosine, 0.999)

        exact = (y.view(torch.uint8) == y_ref.view(torch.uint8)).float().mean().item()
        self.assertGreater(exact, 0.95)

    def test_matches_production_two_pass(self):
        for rows in _ROWS:
            for seed in _SEEDS:
                for scale_fmt in (None, "ue8m0"):
                    with self.subTest(rows=rows, seed=seed, scale_fmt=scale_fmt):
                        x = self._input(rows, seed)
                        y, scale = fused_hadamard_act_quant(x, _BLOCK, scale_fmt)
                        y_ref, scale_ref = _production_reference(x, scale_fmt)
                        if scale_fmt == "ue8m0":
                            torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)
                        else:
                            torch.testing.assert_close(
                                scale, scale_ref, rtol=1e-2, atol=0
                            )
                        self._assert_quantized_close(y, scale, y_ref, scale_ref)

    def test_weights_fold(self):
        softmax_scale = _BLOCK**-0.5
        for rows in (1, 64, 4096):
            for seed in _SEEDS:
                with self.subTest(rows=rows, seed=seed):
                    x = self._input(rows, seed)
                    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
                    weights = torch.randn(
                        rows,
                        _HEADS,
                        device="cuda",
                        dtype=torch.float32,
                        generator=generator,
                    )
                    y, scale, scaled_weights = fused_hadamard_act_quant(
                        x,
                        _BLOCK,
                        "ue8m0",
                        weights=weights,
                        softmax_scale=softmax_scale,
                    )
                    expected = weights.unsqueeze(-1) * scale * softmax_scale
                    torch.testing.assert_close(
                        scaled_weights, expected, rtol=1e-6, atol=1e-7
                    )
                    self.assertTrue(torch.isfinite(y.float()).all().item())
                    self.assertTrue(torch.isfinite(scaled_weights).all().item())

    def test_cuda_graph_replay(self):
        x = self._input(64, 7)
        weights = torch.randn(
            64, _HEADS, device="cuda", dtype=torch.float32
        ).contiguous()
        graph = torch.cuda.CUDAGraph()

        fused_hadamard_act_quant(x, _BLOCK, "ue8m0", weights=weights)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            y, scale, scaled_weights = fused_hadamard_act_quant(
                x, _BLOCK, "ue8m0", weights=weights
            )

        expected = None
        for seed in _SEEDS:
            x.copy_(self._input(64, seed))
            graph.replay()
            torch.cuda.synchronize()
            current = (y.clone(), scale.clone(), scaled_weights.clone())
            eager = fused_hadamard_act_quant(x, _BLOCK, "ue8m0", weights=weights)
            for actual, reference in zip(current, eager):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            if expected is not None:
                self.assertFalse(torch.equal(current[0], expected[0]))
            expected = current


if __name__ == "__main__":
    unittest.main()
