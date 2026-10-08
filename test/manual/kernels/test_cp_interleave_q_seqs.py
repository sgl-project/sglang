"""Exercise eager and graph sequence-length dtypes for interleave CP."""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.cp_split import dsa_cp_interleave_q_seqs_kernel
from sglang.test.test_utils import CustomTestCase


class TestCPInterleaveQSeqs(CustomTestCase):
    def test_sequence_length_dtypes(self):
        lengths = [1024, 3, 13, 0, 8]
        for dtype in (torch.int32, torch.int64):
            for size in (2, 4, 8):
                for rank in range(size):
                    with self.subTest(dtype=dtype, size=size, rank=rank):
                        source = torch.tensor(lengths, device="cuda", dtype=dtype)
                        output = torch.full_like(source, -1)
                        indices = torch.full_like(source, -1)
                        dsa_cp_interleave_q_seqs_kernel[(1,)](
                            source, output, indices, len(lengths), size, rank
                        )
                        expected, request_ids = [], []
                        start = 0
                        for request, length in enumerate(lengths):
                            count = sum(
                                token % size == rank
                                for token in range(start, start + length)
                            )
                            if count:
                                expected.append(count)
                                request_ids.append(request)
                            start += length
                        tail = [-1] * (len(lengths) - len(expected))
                        self.assertEqual(output.tolist(), expected + tail)
                        self.assertEqual(indices.tolist(), request_ids + tail)


if __name__ == "__main__":
    unittest.main()
