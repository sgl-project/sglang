"""Candidate tables must cover speculative rows beyond the 1M context boundary."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.candidate_table import (
    build_sparse_indexer_schedule,
    sort_candidate_blocks,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestCandidateTableBoundary(CustomTestCase):
    def test_speculative_context_boundary(self):
        torch.manual_seed(0)
        page_size = 128
        topk = 2048
        for length in (1048575, 1048576, 1048577, 1048586, 2097152):
            for captured in (False, True):
                with self.subTest(length=length, captured=captured):
                    nblocks = (length + 7) // 8
                    chosen = torch.randperm(nblocks - 1, device="cuda")[: topk - 1]
                    chosen = torch.cat([chosen, chosen.new_tensor([nblocks - 1])])
                    blocks = chosen.to(torch.int32)[None, :].contiguous()
                    expected = blocks.sort(dim=1).values
                    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
                    pages = torch.randperm((length + 127) // 128, device="cuda")
                    pages = pages.to(torch.int32)[None, :].contiguous()
                    request_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
                    physical = torch.empty_like(blocks)
                    # Warm up JIT before capturing; sorting is idempotent.
                    sort_candidate_blocks(
                        blocks, lengths, pages, page_size, out_pages=physical
                    )
                    build_sparse_indexer_schedule(
                        blocks, lengths, pages, page_size, torch.int8, request_ids
                    )
                    if captured:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            sort_candidate_blocks(
                                blocks, lengths, pages, page_size, out_pages=physical
                            )
                        graph.replay()
                    build_sparse_indexer_schedule(
                        blocks, lengths, pages, page_size, torch.int8, request_ids
                    )
                    torch.cuda.synchronize()
                    torch.testing.assert_close(blocks, expected)
                    expected_physical = (
                        pages.gather(1, expected.long() // 16) * 16 + expected % 16
                    )
                    torch.testing.assert_close(physical, expected_physical)


if __name__ == "__main__":
    unittest.main()
