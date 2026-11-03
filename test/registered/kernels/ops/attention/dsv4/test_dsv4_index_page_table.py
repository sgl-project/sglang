"""expand_index_page_table on the GPU is bitwise its torch expression, which the
CPU path still runs."""

import unittest

import torch

from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _expand(page_table: torch.Tensor, bpp: int) -> torch.Tensor:
    return expand_index_page_table(
        page_table, full_page_size=64 * bpp, compress_ratio=1, index_page_size=64
    )


def _table(bs: int, n: int) -> torch.Tensor:
    t = torch.randint(0, 1 << 20, (bs, n), dtype=torch.int32, device="cuda")
    if bs and n >= 8:
        t[:, -3:] = -1
        t[0, :4] = torch.tensor(
            [2**31 - 1, 2**30, 2**29 + 7, -(2**31)], dtype=torch.int32
        )
    return t


class TestExpandIndexPageTable(CustomTestCase):
    def test_matches_torch(self):
        torch.manual_seed(0)
        cases = [(1024, 2049, 4), (1024, 2049, 2), (3, 17, 2), (1, 300, 4)]
        cases += [(0, 10, 4), (5, 9, 3)]  # empty; non-power-of-two keeps torch
        for bs, n, bpp in cases:
            with self.subTest(bs=bs, n=n, bpp=bpp):
                t = _table(bs, n)
                out = _expand(t, bpp)
                self.assertEqual(out.dtype, torch.int32)
                self.assertTrue(torch.equal(out.cpu(), _expand(t.cpu(), bpp)))

    def test_row_strided(self):
        t = _table(16, 33)[::2]
        self.assertTrue(torch.equal(_expand(t, 4).cpu(), _expand(t.cpu(), 4)))


if __name__ == "__main__":
    unittest.main()
