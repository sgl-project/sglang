"""CPU tests for finite FP16 pooling and normalization, without model loading."""

import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(lengths, dimensions=None):
    return SimpleNamespace(
        extend_seq_lens=torch.tensor(lengths, dtype=torch.long),
        dimensions=dimensions,
    )


def _reference(row):
    values = row.tolist()
    denominator = max(math.hypot(*values), 1e-12)
    return torch.tensor([value / denominator for value in values], dtype=row.dtype)


class TestPoolerNormalization(CustomTestCase):
    def _check(self, hidden, lengths, pooling_type, dimensions=None):
        original = hidden.clone()
        rows = []
        for index, sequence in enumerate(hidden.split(lengths)):
            if pooling_type == PoolingType.LAST:
                row = sequence[-1]
            elif pooling_type == PoolingType.CLS:
                row = sequence[0]
            else:
                row = sequence.double().mean(dim=0).to(hidden.dtype)
            if dimensions is not None:
                row = row[: dimensions[index]]
            rows.append(_reference(row))
        result = Pooler(pooling_type, normalize=True)(
            hidden, _batch(lengths, dimensions)
        ).embeddings
        self.assertEqual(
            isinstance(result, list),
            dimensions is not None and len(set(dimensions)) > 1,
        )
        for actual, expected in zip(result, rows):
            self.assertEqual(actual.dtype, hidden.dtype)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected, rtol=0, atol=5e-4)
        torch.testing.assert_close(hidden, original, rtol=0, atol=0)

    def test_zero_vectors(self):
        for pooling_type in PoolingType:
            with self.subTest(pooling_type=pooling_type):
                self._check(
                    torch.zeros(5, 4, dtype=torch.float16), [2, 3], pooling_type
                )

    def test_norm_above_float16_range(self):
        # A single token isolates normalization from mean-pooling accumulation.
        hidden = torch.tensor([[60000, 50000, -40000, 30000]], dtype=torch.float16)
        for pooling_type in PoolingType:
            with self.subTest(pooling_type=pooling_type):
                self._check(hidden, [1], pooling_type)

    def test_subnormal_vectors(self):
        hidden = (
            torch.tensor([[1, 2, -1, 0], [0, -1, 1, 2]], dtype=torch.float16) * 2**-24
        )
        for pooling_type in (PoolingType.LAST, PoolingType.CLS):
            with self.subTest(pooling_type=pooling_type):
                self._check(hidden, [1, 1], pooling_type)

    def test_same_dimensions_can_leave_zero_vectors(self):
        hidden = torch.tensor([[0, 0, 3, 4], [0, 0, -3, 4]], dtype=torch.float16)
        for pooling_type in (PoolingType.LAST, PoolingType.CLS):
            with self.subTest(pooling_type=pooling_type):
                self._check(hidden, [1, 1], pooling_type, [2, 2])

    def test_ragged_dimensions_zero_large_and_tiny(self):
        hidden = torch.tensor(
            [[0, 0, 3, 4], [60000, -50000, 40000, 30000], [2**-24, -(2**-24), 0, 0]],
            dtype=torch.float16,
        )
        for pooling_type in (PoolingType.LAST, PoolingType.CLS):
            with self.subTest(pooling_type=pooling_type):
                self._check(hidden, [1, 1, 1], pooling_type, [2, 3, 4])

    def test_noncontiguous_hidden_states(self):
        hidden = torch.tensor(
            [[60000, 0, 40000, 0, -50000, 0], [0, 1, 0, 1, 0, 1]],
            dtype=torch.float16,
        )[:, ::2]
        self.assertFalse(hidden.is_contiguous())
        self._check(hidden, [1, 1], PoolingType.LAST)

    def test_cosine_retrieval_keeps_the_nearest_direction(self):
        hidden = torch.tensor(
            [[60000, 60000], [60000, -60000], [50000, 50000]], dtype=torch.float16
        )
        embeddings = Pooler(PoolingType.LAST, normalize=True)(
            hidden, _batch([1, 1, 1])
        ).embeddings
        similarities = embeddings[1:].float() @ embeddings[0].float()
        self.assertEqual(similarities.argmax().item(), 1)
        torch.testing.assert_close(
            similarities, torch.tensor([0.0, 1.0]), rtol=0, atol=1e-3
        )

    def test_normalization_disabled_preserves_values(self):
        hidden = torch.tensor(
            [[0, 0, 3, 4], [60000, 50000, -40000, 30000]], dtype=torch.float16
        )
        for dimensions in (None, [2, 2], [2, 3]):
            with self.subTest(dimensions=dimensions):
                result = Pooler(PoolingType.LAST, normalize=False)(
                    hidden, _batch([1, 1], dimensions)
                ).embeddings
                for index, actual in enumerate(result):
                    expected = (
                        hidden[index]
                        if dimensions is None
                        else hidden[index, : dimensions[index]]
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_other_dtypes_keep_the_native_normalization(self):
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            for dimensions in (None, [2, 2], [2, 3]):
                with self.subTest(dtype=dtype, dimensions=dimensions):
                    hidden = torch.tensor([[1, -3, 2, 4], [5, 2, -1, 3]], dtype=dtype)
                    result = Pooler(PoolingType.LAST, normalize=True)(
                        hidden, _batch([1, 1], dimensions)
                    ).embeddings
                    for index, actual in enumerate(result):
                        row = (
                            hidden[index]
                            if dimensions is None
                            else hidden[index, : dimensions[index]]
                        )
                        torch.testing.assert_close(
                            actual, F.normalize(row, p=2, dim=-1), rtol=0, atol=0
                        )

    def test_regular_half_values(self):
        hidden = torch.tensor([[3, 4, 0], [2, -3, 6], [1, 0, -2]], dtype=torch.float16)
        for pooling_type in PoolingType:
            with self.subTest(pooling_type=pooling_type):
                self._check(hidden, [2, 1], pooling_type)


if __name__ == "__main__":
    unittest.main()
