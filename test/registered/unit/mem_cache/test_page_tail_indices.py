"""Unit tests for write_page_tail_indices: the req_to_token row must stay valid
over every page the allocator handed out, not just over the requested tokens."""

import unittest

import torch

from sglang.srt.mem_cache.allocation import write_page_tail_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ROW_WIDTH = 64


def _paged_row(page_ids, length, page_size, dtype):
    # A row as the allocator lays it out: page_ids[k] backs row positions
    # [k*ps, (k+1)*ps), and a slot sits at its own offset inside that page.
    positions = torch.arange(length)
    slots = page_ids[positions // page_size] * page_size + positions % page_size
    return slots.to(dtype)


class TestWritePageTailIndices(CustomTestCase):
    def test_tail_continues_the_last_page(self):
        rtt = torch.zeros((1, _ROW_WIDTH), dtype=torch.int32)
        rtt[0, :5] = torch.arange(40, 45, dtype=torch.int32)

        write_page_tail_indices(rtt, torch.tensor([0]), torch.tensor([5]), 4)

        self.assertEqual(rtt[0, :8].tolist(), [40, 41, 42, 43, 44, 45, 46, 47])

    def test_completes_the_last_page_and_moves_nothing_else(self):
        # The oracle is the allocator's layout -- a slot sits at its own page
        # offset, and the tail shares the page of the last written slot -- not
        # the implementation's own "last index plus one" rule. Checking every
        # row also subsumes empty rows, page-aligned ends and page_size 1.
        generator = torch.Generator().manual_seed(0)
        num_rows = 4
        for page_size in (1, 2, 4, 8, 16):
            num_pages = _ROW_WIDTH // page_size
            for _ in range(50):
                batch_size = int(
                    torch.randint(1, num_rows + 1, (1,), generator=generator)
                )
                req_pool_indices = torch.randperm(num_rows, generator=generator)[
                    :batch_size
                ]
                write_ends = torch.randint(
                    0, _ROW_WIDTH - page_size + 1, (batch_size,), generator=generator
                )
                rtt = torch.randint(
                    0, 1000, (num_rows, _ROW_WIDTH), generator=generator
                ).to(torch.int32)
                for row, end in zip(req_pool_indices.tolist(), write_ends.tolist()):
                    page_ids = torch.randperm(64, generator=generator)[:num_pages]
                    rtt[row, :end] = _paged_row(page_ids, end, page_size, rtt.dtype)
                before = rtt.clone()

                write_page_tail_indices(rtt, req_pool_indices, write_ends, page_size)

                offsets = torch.arange(_ROW_WIDTH, dtype=rtt.dtype) % page_size
                for row, end in zip(req_pool_indices.tolist(), write_ends.tolist()):
                    ceiling = -(-end // page_size) * page_size
                    where = f"{page_size=} {row=} {end=}"
                    self.assertTrue(
                        torch.equal(rtt[row, :ceiling] % page_size, offsets[:ceiling]),
                        where,
                    )
                    if ceiling > end:
                        self.assertTrue(
                            torch.equal(
                                rtt[row, end:ceiling] // page_size,
                                rtt[row, end - 1 : end] // page_size,
                            ),
                            where,
                        )
                    self.assertTrue(torch.equal(rtt[row, :end], before[row, :end]))
                    self.assertTrue(
                        torch.equal(rtt[row, ceiling:], before[row, ceiling:]), where
                    )

                untouched = sorted(
                    set(range(num_rows)) - set(req_pool_indices.tolist())
                )
                self.assertTrue(torch.equal(rtt[untouched], before[untouched]))


if __name__ == "__main__":
    unittest.main()
