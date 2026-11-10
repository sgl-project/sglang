from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.speculative.gather_spec_extras import (
    gather_seq_lens_and_bonus,
)
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestGatherSeqLensAndBonus(CustomTestCase):
    def _check(self, pool_size, indices):
        g = torch.Generator(device="cuda").manual_seed(pool_size + indices.numel())
        seq_buf = torch.randint(
            0, 1 << 40, (pool_size,), dtype=torch.int64, device="cuda", generator=g
        )
        tok_buf = torch.randint(
            0, 1 << 20, (pool_size,), dtype=torch.int64, device="cuda", generator=g
        )
        seq_ref, tok_ref = seq_buf[indices], tok_buf[indices]
        seq_out, tok_out = gather_seq_lens_and_bonus(indices, seq_buf, tok_buf)
        for out, ref in ((seq_out, seq_ref), (tok_out, tok_ref)):
            self.assertEqual(out.dtype, ref.dtype)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.equal(out, ref))
        self.assertTrue(torch.equal(seq_buf[indices], seq_ref))

    def test_matches_index_gather(self):
        for pool_size in (1, 2, 129, 4097):
            for m in (0, 1, 2, 7, 64, 1500):
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(pool_size=pool_size, m=m, dtype=dtype):
                        indices = torch.randint(
                            0, pool_size, (m,), dtype=dtype, device="cuda"
                        )
                        self._check(pool_size, indices)

    def test_strided_indices(self):
        base = torch.arange(0, 64, dtype=torch.int64, device="cuda").flip(0)
        self._check(64, base[::3])

    def test_in_cuda_graph(self):
        pool_size, m = 256, 4
        seq_buf = torch.arange(pool_size, dtype=torch.int64, device="cuda") * 7
        tok_buf = torch.arange(pool_size, dtype=torch.int64, device="cuda") * 3 + 1
        indices = torch.tensor([5, 0, 255, 17], dtype=torch.int64, device="cuda")
        gather_seq_lens_and_bonus(indices, seq_buf, tok_buf)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            seq_out, tok_out = gather_seq_lens_and_bonus(indices, seq_buf, tok_buf)
        for step in range(3):
            indices.copy_(torch.randint(0, pool_size, (m,), device="cuda"))
            seq_buf.add_(step)
            graph.replay()
            self.assertTrue(torch.equal(seq_out, seq_buf[indices]))
            self.assertTrue(torch.equal(tok_out, tok_buf[indices]))


if __name__ == "__main__":
    unittest.main()
