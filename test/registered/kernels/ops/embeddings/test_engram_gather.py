"""Strided hash views must gather identical rows without an index repack."""

import unittest

import torch

from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestEngramGather(CustomTestCase):
    def test_layer_and_transposed_views_match_contiguous_indices(self):
        weight = torch.randn(31, 64, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.full((31, 2), 127, dtype=torch.uint8, device="cuda")
        # This test isolates index strides: every index is inside the table.
        hashes = torch.randint(0, weight.shape[0], (13, 3, 4), device="cuda")
        hashes[0, 1] = torch.tensor([0, 1, 29, 30], device="cuda")
        for dtype in (torch.int32, torch.int64):
            typed = hashes.to(dtype)
            for indices in (
                typed[:, 1].contiguous().reshape(-1),
                typed[:, 1],
                typed[:, 1].T,
                typed[:, 1, 0],
            ):
                expected = (
                    weight.float()[indices.long()].reshape(-1, 64).to(torch.bfloat16)
                )
                actual = torch.empty_like(expected)
                engram_gather(
                    weight.data_ptr(),
                    scale.data_ptr(),
                    indices,
                    actual,
                    64,
                    32,
                    row_hi=weight.shape[0],
                )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
