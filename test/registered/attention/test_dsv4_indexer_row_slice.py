"""Row-slicing equivalence for the DSV4 indexer logits/top-k chunking.

The chunked path computes logits and top-k over query-row slices. This pins the
property that makes that safe: every row of the top-k transform is a pure
function of its own logits row, sequence length and page-table row, so slicing
the batch cannot change any row's output.

CPU-only -- exercises the pure-torch reference transform and the budget
arithmetic, so it needs no GPU.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsv4.indexer import (
    _mqa_logits_chunk_rows,
    _mqa_logits_row_bytes,
    topk_transform_512_pytorch_vectorized,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")

PAGE_SIZE = 64
TOPK = 512
_BUDGET_FN = "sglang.srt.layers.attention.dsv4.indexer._mqa_logits_budget_bytes"


class TestDsv4IndexerRowSlice(unittest.TestCase):
    def _run(self, logits, seq_lens, page_table, rows):
        out = torch.full((rows, TOPK), -1, dtype=torch.int32)
        topk_transform_512_pytorch_vectorized(
            logits, seq_lens, page_table, out, PAGE_SIZE, None
        )
        return out

    def test_chunked_topk_matches_unchunked(self):
        torch.manual_seed(0)
        rows, width = 37, 2048
        logits = torch.randn(rows, width, dtype=torch.float32)
        seq_lens = torch.randint(1, width, (rows,), dtype=torch.int32)
        page_table = torch.randint(
            0, 4096, (rows, (width + PAGE_SIZE - 1) // PAGE_SIZE), dtype=torch.int32
        )
        expected = self._run(logits, seq_lens, page_table, rows)

        for chunk in (1, 7, 16, rows - 1, rows):
            got = torch.full((rows, TOPK), -1, dtype=torch.int32)
            for start in range(0, rows, chunk):
                end = min(start + chunk, rows)
                topk_transform_512_pytorch_vectorized(
                    logits[start:end],
                    seq_lens[start:end],
                    page_table[start:end],
                    got[start:end],
                    PAGE_SIZE,
                    None,
                )
            self.assertTrue(torch.equal(got, expected), f"chunk={chunk} diverged")

    def test_row_bytes_matches_deepgemm_alignment(self):
        # DeepGEMM pads the fp32 logits row stride to 256 columns.
        self.assertEqual(_mqa_logits_row_bytes(1), 256 * 4)
        self.assertEqual(_mqa_logits_row_bytes(256), 256 * 4)
        self.assertEqual(_mqa_logits_row_bytes(257), 512 * 4)

    def test_no_chunking_without_measurable_budget(self):
        with mock.patch(_BUDGET_FN, return_value=None):
            self.assertIsNone(
                _mqa_logits_chunk_rows(1 << 20, 1 << 20, torch.device("cpu"))
            )

    def test_chunk_rows_slices_when_over_budget(self):
        budget = 512 << 20
        with mock.patch(_BUDGET_FN, return_value=budget):
            # 4096 rows x 93184 cols fp32 = 1.42 GiB > 512 MiB -> must slice.
            chunk = _mqa_logits_chunk_rows(4096, 93184, torch.device("cpu"))
            self.assertIsNotNone(chunk)
            self.assertLess(chunk, 4096)
            self.assertLessEqual(chunk * _mqa_logits_row_bytes(93184), budget)
            # Within budget -> no slicing.
            self.assertIsNone(_mqa_logits_chunk_rows(64, 93184, torch.device("cpu")))


if __name__ == "__main__":
    unittest.main()
