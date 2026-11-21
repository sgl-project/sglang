import unittest

import torch

from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

CHUNK_SIZE = 64


class TestFlaChunkIndices(CustomTestCase):
    def test_empty_sequences_keep_their_ids(self):
        """A zero-length sequence owns no chunk but keeps its id.
        Later sequences' chunks must not take it over.
        The row index of chunk i of sequence s must equal chunk_offsets[s] + i."""
        cases = ([5, 0, 70], [0, 70], [64, 0, 0, 66], [70, 0], [1, 64, 65], [0, 0])
        for dtype in (torch.int32, torch.int64):
            for seq_lens in cases:
                with self.subTest(seq_lens=seq_lens, dtype=dtype):
                    cu_seqlens = torch.tensor([0, *seq_lens]).cumsum(0, dtype=dtype)
                    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
                    chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)

                    expected = [
                        [seq, chunk]
                        for seq, seq_len in enumerate(seq_lens)
                        for chunk in range(-(-seq_len // CHUNK_SIZE))
                    ]
                    self.assertEqual(chunk_indices.dtype, cu_seqlens.dtype)
                    self.assertEqual(chunk_indices.device, cu_seqlens.device)
                    self.assertEqual(chunk_indices.shape, (len(expected), 2))
                    self.assertEqual(chunk_indices.tolist(), expected)
                    for row, (seq, chunk) in enumerate(expected):
                        self.assertEqual(chunk_offsets[seq].item() + chunk, row)


if __name__ == "__main__":
    unittest.main()
