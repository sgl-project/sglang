"""Row-split k-pool indexer prefill: every rank scores a share of the query rows and
the shares are all-gathered. The gathered top-k must select what one rank scoring
every row selects.

Ranks are simulated on one GPU with the real DeepGEMM logits and fused top-k
kernels. The radix selector breaks score ties arbitrarily, so pooled columns are
compared by the scores they select; tail columns are positional and compared exactly.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

KPOOL, TOPK, HEADS, HEAD_DIM = 4, 2048, 32, 128


class _FakeGroup:
    def __init__(self, rank, world_size, shares):
        self.rank_in_group = rank
        self.world_size = world_size
        self.shares = shares

    def all_gather_into_tensor(self, output, input):
        self.shares[self.rank_in_group] = input.clone()
        if len(self.shares) == self.world_size:
            output.copy_(torch.cat([self.shares[r] for r in range(self.world_size)]))


def _indexer():
    indexer = IndexerKPool.__new__(IndexerKPool)
    indexer.index_kpool = KPOOL
    indexer.index_topk = TOPK
    return indexer


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
    "Requires a DeepGEMM-capable GPU",
)
class TestKpoolIndexerRowSplit(CustomTestCase):
    def _inputs(self, rows, groups):
        torch.manual_seed(0)
        dev = "cuda"
        q = torch.randn(rows, HEADS, HEAD_DIM, device=dev).to(torch.float8_e4m3fn)
        k = torch.randn(groups, HEAD_DIM, device=dev).to(torch.float8_e4m3fn)
        k_scale = torch.rand(groups, device=dev) + 0.5
        weights = torch.randn(rows, HEADS, device=dev)
        # Some rows hold fewer valid groups than group_topk, leaving -1 columns.
        pool_lens = torch.randint(
            256, groups + 1, (rows,), device=dev, dtype=torch.int32
        )
        seq_lens = pool_lens * KPOOL + torch.randint(
            0, KPOOL, (rows,), device=dev, dtype=torch.int32
        )
        return dict(
            q_fp8=q,
            weights=weights,
            kv_fp8=(k, k_scale),
            logits_starts=torch.zeros(rows, device=dev, dtype=torch.int32),
            logits_ends=pool_lens,
            pool_lens=pool_lens,
            seq_lens=seq_lens,
            page_table=None,
            page_table_row_index=None,
            topk_offsets=None,
            topk_row_starts=torch.zeros(rows, device=dev, dtype=torch.int32),
        )

    def _selected_scores(self, logits, out):
        pooled = out[:, :TOPK].long()
        scores = logits.gather(1, (pooled.clamp(min=0) // KPOOL))
        scores = torch.where(
            pooled >= 0, scores, torch.full_like(scores, -float("inf"))
        )
        return scores.sort(dim=1).values

    def _check(self, rows, groups, row_chunks, out_rows, world_size=4):
        indexer = _indexer()
        inputs = self._inputs(rows, groups)
        with mock.patch.object(
            dsa_indexer_kpool, "_indexer_row_split_group", lambda **_: None
        ):
            ref = indexer._kpool_topk_by_row_chunks(
                **inputs,
                row_chunks=(slice(0, rows),),
                out_rows=out_rows,
                logits_elems=rows * groups,
            )

        shares = {}

        def run(rank):
            group = _FakeGroup(rank, world_size, shares)
            with mock.patch.object(
                dsa_indexer_kpool, "_indexer_row_split_group", lambda **_: group
            ):
                return indexer._kpool_topk_by_row_chunks(
                    **inputs,
                    row_chunks=row_chunks,
                    out_rows=out_rows,
                    logits_elems=rows * groups,
                )

        # The first pass collects every rank's share; the second gathers them.
        for rank in range(world_size):
            run(rank)
        results = [run(rank) for rank in range(world_size)]

        k, k_scale = inputs["kv_fp8"]
        logits = IndexerKPool._fp8_mqa_logits(
            q_fp8=inputs["q_fp8"],
            k_fp8=k,
            k_scale=k_scale,
            weights=inputs["weights"],
            starts=inputs["logits_starts"],
            ends=inputs["logits_ends"],
            clean_logits=True,
        )
        expected_rows = rows if out_rows is None else out_rows
        for rank, out in enumerate(results):
            with self.subTest(rank=rank):
                self.assertEqual(tuple(out.shape), tuple(ref.shape))
                self.assertEqual(out.shape[0], expected_rows)
                self.assertTrue(torch.equal(out[:, TOPK:], ref[:, TOPK:]))
                self.assertTrue(torch.all(out[rows:] == -1))
                self.assertTrue(
                    torch.equal(
                        self._selected_scores(logits, out[:rows]),
                        self._selected_scores(logits, ref[:rows]),
                    )
                )

    def test_single_chunk(self):
        self._check(rows=512, groups=2048, row_chunks=(slice(0, 512),), out_rows=None)

    def test_budget_chunks_straddle_rank_shares(self):
        # 515 rows give shares of 129 with a short last share; the budget chunks
        # cut across share boundaries, and out_rows adds padding rows.
        self._check(
            rows=515,
            groups=2048,
            row_chunks=(slice(0, 200), slice(200, 400), slice(400, 515)),
            out_rows=520,
        )

    def test_eight_ranks_with_uneven_shares(self):
        # 33 rows over 8 ranks: ceil-sized shares would leave the last rank none.
        self._check(
            rows=33,
            groups=2048,
            row_chunks=(slice(0, 33),),
            out_rows=None,
            world_size=8,
        )


if __name__ == "__main__":
    unittest.main()
