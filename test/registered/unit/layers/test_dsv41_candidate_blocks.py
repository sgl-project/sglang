import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    select_candidate_block_ids,
    topk_among_blocks,
)
from sglang.srt.layers.attention.dsv4 import v41_indexer
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
    BlockIds,
    DenseBlocksBackend,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.types import get_tail_row_indices
from sglang.srt.layers.deep_gemm_wrapper import configurer
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def id_sets(blocks: torch.Tensor) -> list:
    return [set(row.tolist()) - {-1} for row in blocks]


def masked_topk(scores, lens, blocks, k, block_size):
    """The reference: -inf outside the chosen blocks and past each row's length,
    then a plain top-k; -1 where fewer than k candidates are finite."""
    rows, width = scores.shape
    keep = torch.zeros(rows, width, dtype=torch.bool)
    for r in range(rows):
        for b in blocks[r].tolist():
            if b >= 0:
                keep[r, b * block_size : (b + 1) * block_size] = True
    keep &= torch.arange(width)[None, :] < lens[:, None]
    masked = scores.masked_fill(~keep, -torch.inf)
    top = masked.topk(min(k, width), dim=-1)
    return [
        {c for c, v in zip(ci.tolist(), vi.tolist()) if v > -torch.inf}
        for ci, vi in zip(top.indices, top.values)
    ]


class TestCandidateBlockIds(CustomTestCase):
    def test_causal_partial_blocks_and_forced_newest_block(self):
        scores = torch.tensor([[100.0] * 8 + [50.0] * 8 + [-10.0] * 3] * 4)
        lengths = torch.tensor([[0], [1], [9], [19]])
        scores.masked_fill_(torch.arange(19)[None, :] >= lengths, -torch.inf)
        original_scores = scores.clone()
        blocks = select_candidate_block_ids(
            logits=scores, compress_lens=lengths, topk_blocks=2, block_size=8
        )
        self.assertEqual(blocks.dtype, torch.int32)
        self.assertEqual(tuple(blocks.shape), (4, 2))
        # nothing visible; one partial block; two blocks; the newest (partial)
        # block is forced in ahead of the 50-score one
        self.assertEqual(id_sets(blocks), [set(), {0}, {0, 1}, {0, 2}])
        torch.testing.assert_close(scores, original_scores)

    def test_underfilled_candidates_keep_every_block(self):
        for width in (0, 1, 7, 8, 9):
            with self.subTest(width=width):
                blocks = select_candidate_block_ids(
                    logits=torch.zeros((2, width)),
                    compress_lens=width,
                    topk_blocks=2048,
                    block_size=8,
                )
                expected = set(range((width + 7) // 8))
                self.assertEqual(id_sets(blocks), [expected, expected])


class TestTopkAmongBlocks(CustomTestCase):
    def test_matches_masked_topk(self):
        """Gathering the chosen blocks and taking the top-k picks the same
        positions as masking everything else to -inf, including causal cuts,
        -1 padded blocks and rows with fewer than k candidates."""
        g = torch.Generator().manual_seed(0)
        for rows, width, topk_blocks, k in (
            (7, 100, 3, 16),
            (5, 1000, 20, 64),
            (3, 13, 4, 32),
            (2, 8, 1, 4),
        ):
            with self.subTest(width=width, k=k):
                scores = torch.randn(rows, width, generator=g)
                lens = torch.randint(0, width + 1, (rows,), generator=g)
                visible = scores.masked_fill(
                    torch.arange(width)[None, :] >= lens[:, None], -torch.inf
                )
                blocks = select_candidate_block_ids(
                    visible, lens[:, None], topk_blocks=topk_blocks, block_size=8
                )
                picked = topk_among_blocks(scores, lens, blocks, k, block_size=8)
                self.assertEqual(tuple(picked.shape), (rows, k))
                self.assertEqual(
                    id_sets(picked), masked_topk(scores, lens, blocks, k, 8)
                )

    def test_no_candidate_selects_nothing(self):
        scores = torch.randn(2, 16)
        blocks = torch.full((2, 2), -1, dtype=torch.int32)
        picked = topk_among_blocks(scores, torch.tensor([16, 0]), blocks, 4, 8)
        self.assertTrue((picked == -1).all())


class TestBlockIdsTail(CustomTestCase):
    def test_tail_keeps_each_requests_last_rows(self):
        blocks = torch.arange(16, dtype=torch.int32).view(8, 2)
        ids = BlockIds(blocks=blocks, rows_per_request=[5, 0, 3])
        tail = ids.tail([2, 0, 0])
        torch.testing.assert_close(tail.blocks, blocks[3:5])
        self.assertEqual(tail.rows_per_request, [2, 0, 0])
        self.assertEqual(
            get_tail_row_indices(
                full_rows_per_request=[5, 0, 3],
                tail_rows_per_request=[2, 0, 3],
                device="cpu",
            ).tolist(),
            [3, 4, 5, 6, 7],
        )


class TestCandidateIndexerGating(CustomTestCase):
    """SM12x has no DeepGEMM paged sparse MQA logits; backend init must give it
    the dense-blocks scheme, not fail, while SM100 without them still fails."""

    def _make_without_sparse_logits(self, **platform):
        with (
            patch.object(v41_indexer, "is_sm100_or_newer", return_value=True),
            patch.object(v41_indexer, "_use_deep_gemm_prefill", return_value=True),
            patch.object(configurer, "DEEPGEMM_PAGED_SPARSE_MQA_LOGITS", False),
            override_platform(**platform),
        ):
            return v41_indexer.make_candidate_indexer(
                token_to_kv_pool=None,
                req_to_token=None,
                page_size=64,
                candidate_topk_blocks=4,
                candidate_block_size=8,
            )

    def test_sm12x_takes_dense_blocks(self):
        prefill, decode = self._make_without_sparse_logits(is_sm120=True)
        self.assertIsInstance(decode, DenseBlocksBackend)
        self.assertIs(prefill, decode)

    def test_sm100_still_requires_sparse_logits(self):
        with self.assertRaisesRegex(RuntimeError, "paged sparse MQA logits"):
            self._make_without_sparse_logits(is_sm120=False)


if __name__ == "__main__":
    unittest.main()
