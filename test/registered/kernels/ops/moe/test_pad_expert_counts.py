import unittest

import torch

from sglang.kernels.ops.moe.triton_pad_expert_counts import pad_expert_counts
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "pad_expert_counts needs CUDA")
class TestPadExpertCounts(CustomTestCase):
    def test_matches_eager(self):
        cases = (
            ([0], 16, 32),
            ([1, 8, 9, 0], 8, 48),
            ([0, 1, 7, 8, 9, 31, 32], 16, 160),
        )
        for dtype in (torch.int32, torch.int64):
            for values, block_e, all_tokens in cases:
                with self.subTest(dtype=dtype, values=values):
                    counts = torch.tensor(values, device="cuda", dtype=dtype)
                    expected = (((counts + block_e - 1) // block_e) * block_e).to(
                        torch.int32
                    )
                    expected[-1].add_(all_tokens - expected.sum())

                    actual = pad_expert_counts(counts, block_e, all_tokens)

                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.dtype, torch.int32)
                    self.assertEqual(actual.sum().item(), all_tokens)


if __name__ == "__main__":
    unittest.main()
