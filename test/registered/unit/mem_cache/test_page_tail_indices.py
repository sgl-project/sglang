"""Unit tests for write_page_tail_indices: the req_to_token row must stay valid
over every page the allocator handed out, not just over the requested tokens."""

import unittest

import torch

from sglang.srt.mem_cache.allocation import write_page_tail_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ROW_WIDTH = 64


def _reference(rtt, req_pool_indices, write_ends, page_size):
    out = rtt.clone()
    for row, end in zip(req_pool_indices.tolist(), write_ends.tolist()):
        if page_size == 1 or end == 0:
            continue
        ceiling = -(-end // page_size) * page_size
        for step, pos in enumerate(range(end, ceiling), start=1):
            out[row, pos] = rtt[row, end - 1] + step
    return out


class TestWritePageTailIndices(CustomTestCase):
    def test_tail_continues_the_last_page(self):
        rtt = torch.zeros((1, _ROW_WIDTH), dtype=torch.int32)
        rtt[0, :5] = torch.arange(40, 45, dtype=torch.int32)

        write_page_tail_indices(rtt, torch.tensor([0]), torch.tensor([5]), 4)

        self.assertEqual(rtt[0, :8].tolist(), [40, 41, 42, 43, 44, 45, 46, 47])

    def test_matches_naive_reference_over_random_batches(self):
        # Comparing the whole pool subsumes page-aligned ends, empty rows,
        # page_size 1 and untouched rows -- no separate case needed for those.
        generator = torch.Generator().manual_seed(0)
        num_rows = 4
        for page_size in (1, 2, 4, 8, 16):
            for _ in range(50):
                batch_size = int(
                    torch.randint(1, num_rows + 1, (1,), generator=generator)
                )
                req_pool_indices = torch.randperm(num_rows, generator=generator)[
                    :batch_size
                ]
                write_ends = torch.randint(
                    0, _ROW_WIDTH - page_size, (batch_size,), generator=generator
                )
                rtt = torch.randint(
                    0, 1000, (num_rows, _ROW_WIDTH), generator=generator
                ).to(torch.int32)
                for row, end in zip(req_pool_indices.tolist(), write_ends.tolist()):
                    base = (
                        int(torch.randint(0, 8, (1,), generator=generator)) * page_size
                    )
                    rtt[row, :end] = torch.arange(base, base + end, dtype=rtt.dtype)

                expected = _reference(rtt, req_pool_indices, write_ends, page_size)
                write_page_tail_indices(rtt, req_pool_indices, write_ends, page_size)

                self.assertTrue(
                    torch.equal(rtt, expected),
                    f"{page_size=} {req_pool_indices=} {write_ends=}",
                )


if __name__ == "__main__":
    unittest.main()
