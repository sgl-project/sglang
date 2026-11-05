import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPadTopkIndicesOffset(CustomTestCase):
    def test_both_indices_pad_to_q_len(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        topk = 4
        num_tokens = 5

        topk_indices = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, -1, -1],
                [5, 6, 7, -1],
            ],
            dtype=torch.int32,
        )
        topk_indices_offset = torch.tensor([0, 100, 200], dtype=torch.int32)

        padded_indices = backend._pad_topk_indices(topk_indices, num_tokens)
        padded_offset = backend._pad_topk_indices_offset(
            topk_indices_offset, num_tokens
        )

        self.assertEqual(padded_indices.shape, (num_tokens, topk))
        self.assertEqual(padded_offset.ndim, 1)
        self.assertEqual(padded_offset.shape, (num_tokens,))
        self.assertTrue(
            torch.equal(
                padded_indices,
                torch.tensor(
                    [
                        [0, 1, 2, 3],
                        [0, 1, -1, -1],
                        [5, 6, 7, -1],
                        [-1, -1, -1, -1],
                        [-1, -1, -1, -1],
                    ],
                    dtype=torch.int32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                padded_offset,
                torch.tensor([0, 100, 200, 0, 0], dtype=torch.int32),
            )
        )

        mask = padded_indices != -1
        transformed = torch.where(
            mask, padded_indices + padded_offset.unsqueeze(1), padded_indices
        )
        expected = torch.tensor(
            [
                [0, 1, 2, 3],
                [100, 101, -1, -1],
                [205, 206, 207, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(transformed, expected))

    def test_only_offset_indices_pad_to_q_len(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        topk = 4
        num_tokens = 5

        topk_indices = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, -1, -1],
                [5, 6, 7, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        topk_indices_offset = torch.tensor([0, 100, 200], dtype=torch.int32)

        padded_indices = backend._pad_topk_indices(topk_indices, num_tokens)
        padded_offset = backend._pad_topk_indices_offset(
            topk_indices_offset, num_tokens
        )

        self.assertIs(padded_indices, topk_indices)
        self.assertEqual(padded_indices.shape, (num_tokens, topk))
        self.assertEqual(padded_offset.shape, (num_tokens,))
        self.assertTrue(
            torch.equal(
                padded_offset,
                torch.tensor([0, 100, 200, 0, 0], dtype=torch.int32),
            )
        )

        mask = padded_indices != -1
        transformed = torch.where(
            mask, padded_indices + padded_offset.unsqueeze(1), padded_indices
        )
        expected = torch.tensor(
            [
                [0, 1, 2, 3],
                [100, 101, -1, -1],
                [205, 206, 207, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(transformed, expected))


if __name__ == "__main__":
    unittest.main()
