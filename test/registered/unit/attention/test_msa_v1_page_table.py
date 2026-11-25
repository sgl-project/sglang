# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for ``build_msa_v1_page_table`` (MiniMax-M3 MSA-v1).

The page table must hold, for every request row, the 128-token page id of each
logical page up to ``max_seqlen_k``. It is built with one 2-D gather so that a
large batch (the FlashInfer extend-autotune dummy batch at a 1M context) does
not copy every request's full ``req_to_token`` row first.
"""

import math
import unittest

import torch

from sglang.srt.layers.attention.minimax_sparse_ops.msa_plan import (
    build_msa_v1_page_table,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE = 128


def _reference(req_to_token, req_pool_indices, max_seqlen_k):
    pages = max(1, math.ceil(max_seqlen_k / _PAGE))
    out = torch.empty((req_pool_indices.numel(), pages), dtype=torch.int32)
    for r, row in enumerate(req_pool_indices.tolist()):
        for p in range(pages):
            out[r, p] = int(req_to_token[row, p * _PAGE]) // _PAGE
    return out


class TestBuildMsaV1PageTable(CustomTestCase):
    def test_matches_reference(self):
        g = torch.Generator().manual_seed(0)
        for dtype in (torch.int32, torch.int64):
            for rows, capacity, max_seqlen_k in [
                (1, 256, 1),
                (3, 4096, 129),
                (5, 8192, 8064),
                (7, 2048, 2048),
            ]:
                with self.subTest(dtype=dtype, rows=rows, max_seqlen_k=max_seqlen_k):
                    table = torch.randint(
                        0, 1 << 30, (16, capacity), generator=g, dtype=torch.int64
                    ).to(dtype)
                    idx = torch.randint(0, 16, (rows,), generator=g, dtype=torch.int32)
                    got = build_msa_v1_page_table(table, idx, max_seqlen_k=max_seqlen_k)
                    self.assertEqual(got.dtype, torch.int32)
                    self.assertTrue(got.is_contiguous())
                    torch.testing.assert_close(
                        got, _reference(table, idx, max_seqlen_k), rtol=0, atol=0
                    )

    def test_repeated_rows_and_full_width(self):
        # 8,192 pages is the MSA-v1 ceiling (1M tokens); many dummy rows share one pool row.
        table = torch.arange(4 * (8192 * _PAGE + _PAGE), dtype=torch.int32).view(4, -1)
        idx = torch.zeros(64, dtype=torch.int32)
        idx[1::2] = 3
        got = build_msa_v1_page_table(table, idx, max_seqlen_k=8192 * _PAGE)
        self.assertEqual(tuple(got.shape), (64, 8192))
        cols = torch.arange(8192) * _PAGE
        torch.testing.assert_close(got[0], (table[0, cols] // _PAGE).to(torch.int32))
        torch.testing.assert_close(got[1], (table[3, cols] // _PAGE).to(torch.int32))

    def test_rejects_too_many_pages(self):
        table = torch.zeros((1, 8193 * _PAGE + 1), dtype=torch.int32)
        with self.assertRaises(ValueError):
            build_msa_v1_page_table(
                table, torch.zeros(1, dtype=torch.int32), max_seqlen_k=8193 * _PAGE
            )


if __name__ == "__main__":
    unittest.main()
