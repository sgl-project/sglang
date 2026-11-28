"""Unit tests for rebuilding a DSA draft page table after DP-attention padding."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa_backend import (
    rebuild_padded_draft_page_table,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPaddedDraftPageTable(CustomTestCase):
    def test_rebuild_extends_to_padded_rows(self):
        # Planned metadata for 1 real request; DP padding made topk 2 rows.
        req_to_token = torch.arange(24, dtype=torch.int64).reshape(2, 12)
        pool = SimpleNamespace(req_to_token=req_to_token)
        metadata = SimpleNamespace(page_table_1=torch.ones(1, 4, dtype=torch.int64))
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 0]),
            seq_lens=torch.tensor([5, 3]),
        )
        topk_indices = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int64)

        rebuilt = rebuild_padded_draft_page_table(
            metadata, forward_batch, topk_indices, pool
        )

        self.assertEqual(rebuilt.shape, (2, 6))
        self.assertTrue(
            torch.equal(rebuilt[0], req_to_token[1, :6]),
            "row 0 must map request 1's real tokens",
        )
        # The dummy padded row maps request 0's table row; its topk is all -1
        # so the downstream gather is inert for it.
        self.assertTrue(torch.equal(rebuilt[1], req_to_token[0, :6]))
        self.assertIs(metadata.page_table_1, rebuilt)

    def test_rebuild_preserves_planned_width(self):
        # Page table wider than max_seqlen+1 keeps the planned width.
        req_to_token = torch.arange(18, dtype=torch.int64).reshape(1, 18)
        pool = SimpleNamespace(req_to_token=req_to_token)
        metadata = SimpleNamespace(page_table_1=torch.zeros(1, 9, dtype=torch.int64))
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([2]),
        )
        rebuilt = rebuild_padded_draft_page_table(
            metadata, forward_batch, torch.zeros(1, 2, dtype=torch.int64), pool
        )
        self.assertEqual(rebuilt.shape, (1, 9))


if __name__ == "__main__":
    unittest.main()
