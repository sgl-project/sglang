"""Bit-exact CUDA coverage for the BF16 prefill indexer score epilogue."""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention.dsv4.index_score_reduce import (
    _reduce_32_heads,
    reduce_head_scores,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestReduceHeadScoresCPU(unittest.TestCase):
    def test_cpu_fallback(self):
        gen = torch.Generator().manual_seed(42)
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                scores = torch.randn(3, 32, 257, dtype=dtype, generator=gen)
                weights = torch.randn(3, 32, dtype=dtype, generator=gen)
                expected = (scores.relu() * weights.unsqueeze(-1)).sum(1).float()
                with mock.patch.object(
                    _reduce_32_heads, "run", side_effect=AssertionError("CUDA launch")
                ):
                    actual = reduce_head_scores(scores, weights)
                self.assertTrue(torch.equal(actual, expected))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "requires NVIDIA CUDA"
)
class TestReduceHeadScoresCUDA(unittest.TestCase):
    def _check(self, scores, weights, *, launches=True):
        expected = (scores.relu() * weights.unsqueeze(-1)).sum(1).float()
        with mock.patch.object(
            _reduce_32_heads, "run", wraps=_reduce_32_heads.run
        ) as run:
            actual = reduce_head_scores(scores, weights)
        self.assertEqual(run.call_count, int(launches))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(
            torch.equal(actual, expected),
            f"shape={tuple(scores.shape)}, strides={scores.stride()}",
        )
        return actual

    def test_fused_shapes(self):
        for rows in (1, 3):
            for width in (2, 3, 31, 255, 256, 257, 511, 2049, 65537):
                for seed in (0, 42, 2026):
                    with self.subTest(rows=rows, width=width, seed=seed):
                        gen = torch.Generator(device="cuda").manual_seed(seed)
                        scores = torch.randn(
                            rows,
                            32,
                            width,
                            device="cuda",
                            dtype=torch.bfloat16,
                            generator=gen,
                        )
                        weights = torch.randn(
                            rows,
                            32,
                            device="cuda",
                            dtype=torch.bfloat16,
                            generator=gen,
                        )
                        self._check(scores, weights)

    def test_non_contiguous_inputs(self):
        for width in (255, 257, 2049):
            gen = torch.Generator(device="cuda").manual_seed(width)
            weights = torch.randn(
                6, 64, device="cuda", dtype=torch.bfloat16, generator=gen
            )[::2, ::2]
            sliced = torch.randn(
                6,
                64,
                width * 2,
                device="cuda",
                dtype=torch.bfloat16,
                generator=gen,
            )[::2, ::2, ::2]
            transposed = torch.randn(
                32,
                3,
                width,
                device="cuda",
                dtype=torch.bfloat16,
                generator=gen,
            ).transpose(0, 1)
            self.assertFalse(weights.is_contiguous())
            for scores in (sliced, transposed):
                with self.subTest(width=width, strides=scores.stride()):
                    self.assertFalse(scores.is_contiguous())
                    self._check(scores, weights)

    def test_bf16_halfway_rounding(self):
        scores = torch.ones(8, 32, 257, device="cuda", dtype=torch.bfloat16)
        weights = torch.zeros(8, 32, device="cuda", dtype=torch.bfloat16)
        # Exact halfway sums: even and odd BF16 mantissas, with both signs.
        weights[0, :2] = torch.tensor([1.0, 2**-8], device="cuda")
        weights[1, :2] = torch.tensor([1.0 + 2**-7, 2**-8], device="cuda")
        weights[2] = -weights[0]
        weights[3] = -weights[1]
        # The product is halfway too; omitting its BF16 cast changes the sum.
        scores[4:6, 0] = 1.0 + 2**-7
        weights[4, :2] = torch.tensor([1.5, -(2**-9)], device="cuda")
        weights[5] = -weights[4]
        # Cancellation exposes a different head accumulation tree.
        weights[6, :4] = torch.tensor([2**24, -(2**24), 1.0, 2**-8], device="cuda")
        weights[7] = -weights[6]
        actual = self._check(scores, weights)
        expected = torch.tensor(
            [1.0, 1.015625, -1.0, -1.015625, 1.515625, -1.515625, 1.0, -1.0],
            device="cuda",
        )
        self.assertTrue(torch.equal(actual, expected[:, None].expand_as(actual)))

    @staticmethod
    def _cancellation_weights():
        weights = torch.zeros(2, 32, device="cuda", dtype=torch.bfloat16)
        weights[0, :4] = torch.tensor([2**24, -(2**24), 1.0, 2**-8], device="cuda")
        weights[1] = -weights[0]
        return weights

    def test_head_contiguous_fallback(self):
        scores = torch.ones(2, 257, 32, device="cuda", dtype=torch.bfloat16).transpose(
            1, 2
        )
        self.assertFalse(scores.is_contiguous())
        actual = self._check(scores, self._cancellation_weights(), launches=False)
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_zero_and_one_width_fallback(self):
        weights = self._cancellation_weights()
        for width in (0, 1):
            with self.subTest(width=width):
                scores = torch.ones(2, 32, width, device="cuda", dtype=torch.bfloat16)
                self._check(scores, weights, launches=False)

    def test_unsupported_heads_and_dtypes(self):
        for heads, score_dtype, weight_dtype in (
            (16, torch.bfloat16, torch.bfloat16),
            (33, torch.bfloat16, torch.bfloat16),
            (64, torch.bfloat16, torch.bfloat16),
            (32, torch.float16, torch.float16),
            (32, torch.float32, torch.float32),
            (32, torch.bfloat16, torch.float32),
        ):
            with self.subTest(
                heads=heads, score_dtype=score_dtype, weight_dtype=weight_dtype
            ):
                scores = torch.randn(3, heads, 257, device="cuda", dtype=score_dtype)
                weights = torch.randn(3, heads, device="cuda", dtype=weight_dtype)
                self._check(scores, weights, launches=False)

    def test_empty_rows(self):
        scores = torch.empty(0, 32, 257, device="cuda", dtype=torch.bfloat16)
        weights = torch.empty(0, 32, device="cuda", dtype=torch.bfloat16)
        self._check(scores, weights, launches=False)


if __name__ == "__main__":
    unittest.main()
