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


if __name__ == "__main__":
    unittest.main()
