"""CUDA round-trip test for address-block gather and scatter."""

import unittest

import torch

from sglang.srt.disaggregation.common.staging_buffer import (
    StagingBuffer,
    gather_address_blocks_to_staging,
    scatter_address_blocks_from_staging,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestDsv4StagingKernel(CustomTestCase):
    def test_gather_scatter_round_trip(self):
        src = (torch.arange(4096, device="cuda", dtype=torch.int64) % 251).to(
            torch.uint8
        )
        dst = torch.zeros_like(src)
        staging = StagingBuffer(1 << 20, "cuda:0", 0)
        blocks = [
            (src.data_ptr() + 10, dst.data_ptr() + 100, 2050),
            (src.data_ptr() + 3000, dst.data_ptr() + 3000, 400),
        ]

        _, num_fragments = gather_address_blocks_to_staging(blocks, staging, 0)
        scatter_address_blocks_from_staging(staging.buffer, num_fragments)
        torch.cuda.synchronize()

        self.assertEqual(num_fragments, 4)
        self.assertTrue(torch.equal(dst[100:2150], src[10:2060]))
        self.assertTrue(torch.equal(dst[3000:3400], src[3000:3400]))


if __name__ == "__main__":
    unittest.main()
