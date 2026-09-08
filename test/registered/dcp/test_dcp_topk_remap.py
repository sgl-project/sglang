"""CPU unit test for the DCP sparse top-k remap.

Pins ``remap_dcp_local_topk_indices`` (``layers/dcp/layout.py``), which converts
the sparse indexer's *global* top-k positions into the *rank-local* coordinates
sparse attention reads.

This is P3a's correctness core and the place an off-by-one-shard bug hides. It
survives being a CPU test because the whole thing is index arithmetic: no LSE,
no attention operator, no NPU. If this passes and the phase still fails, the
fault is downstream -- in the operator or the LSE base -- which is exactly the
split the plan draws between P3a and P3b.

The assertions are partition properties over *all* ranks together, not
single-rank examples. A remap that drops a position, double-counts one, or lands
it one shard over satisfies every per-rank spot check you can write and still
loses tokens; only the union catches it.

Usage:
    python -m pytest test_dcp_topk_remap.py -v
    python test_dcp_topk_remap.py
"""

import unittest

import torch

from sglang.srt.layers.dcp.layout import remap_dcp_local_topk_indices
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DCP_SIZES = [2, 3, 4, 8, 16]
PAD = -1


def _remap(topk, dcp_size, dcp_rank):
    with get_parallel().override(
        dcp_enabled=dcp_size > 1,
        attn_dcp_size=dcp_size,
        attn_dcp_rank=dcp_rank,
    ):
        return remap_dcp_local_topk_indices(topk)


def _topk_rows(num_rows: int, k: int, seq_len: int, seed: int) -> torch.Tensor:
    """Rows of DISTINCT positions, padded with -1 -- what the indexer returns.

    Top-k entries within a row are distinct by construction: they are the k
    best-scoring positions of one sequence. Sampling with replacement instead
    makes the partition property below unsatisfiable for a reason that has
    nothing to do with the remap -- a repeated position is claimed twice by the
    rank that owns it -- so the shape of the fixture is load-bearing here.
    """
    g = torch.Generator().manual_seed(seed)
    n = min(k, seq_len)
    pad = torch.full((k - n,), PAD, dtype=torch.int32)
    return torch.stack(
        [
            torch.cat([torch.randperm(seq_len, generator=g)[:n].to(torch.int32), pad])
            for _ in range(num_rows)
        ]
    )


def _valid(row: torch.Tensor) -> list:
    """The meaningful prefix of a remapped row: entries before the first pad.

    Compaction guarantees the pads are a suffix, so anything after the first one
    is padding by construction -- and asserting that here would be circular, so
    a separate test checks it directly.
    """
    out = []
    for v in row.tolist():
        if v == PAD:
            break
        out.append(v)
    return out


class TestDcpTopkRemap(CustomTestCase):
    def test_without_dcp_the_indices_are_untouched(self):
        topk = torch.tensor([[7, 3, 11, PAD]], dtype=torch.int32)
        out = _remap(topk, 1, 0)
        self.assertTrue(torch.equal(out, topk))

    def test_the_shape_never_changes(self):
        # The reason this function marks-and-compacts instead of selecting:
        # index_topk is fixed and every kernel downstream depends on it.
        for dcp_size in DCP_SIZES:
            topk = torch.arange(64, dtype=torch.int32).reshape(4, 16)
            for rank in range(dcp_size):
                with self.subTest(dcp_size=dcp_size, rank=rank):
                    out = _remap(topk, dcp_size, rank)
                    self.assertEqual(out.shape, topk.shape)
                    self.assertEqual(out.dtype, topk.dtype)

    def test_the_ranks_partition_the_selected_positions(self):
        """Every selected global position is claimed by exactly one rank, and
        lands on the row that rank's shard actually stores.

        This is the whole contract in one assertion. The reconstruction
        ``local * dcp_size + rank`` is the inverse of the owner rule, so
        recovering the original position proves the divide and the filter agree
        -- the failure mode where a filter is applied without its matching
        divide passes a shape check and fails here.
        """
        for dcp_size in DCP_SIZES:
            for seq_len in (dcp_size, 97, 1024):
                topk = _topk_rows(5, 24, seq_len, seed=17)
                with self.subTest(dcp_size=dcp_size, seq_len=seq_len):
                    claimed = {}
                    for rank in range(dcp_size):
                        out = _remap(topk, dcp_size, rank)
                        for r, row in enumerate(out):
                            for local in _valid(row):
                                pos = local * dcp_size + rank
                                key = (r, pos)
                                self.assertNotIn(
                                    key,
                                    claimed,
                                    f"position {pos} of row {r} claimed by both "
                                    f"rank {claimed.get(key)} and rank {rank}",
                                )
                                claimed[key] = rank

                    for r, row in enumerate(topk):
                        for pos in row.tolist():
                            self.assertEqual(
                                claimed.get((r, int(pos))),
                                int(pos) % dcp_size,
                                f"position {pos} of row {r} went to the wrong rank",
                            )

    def test_the_top_k_order_survives_the_remap(self):
        """Compaction must not reorder. Top-k is ranked output: the indexer put
        the most relevant position first, and the attention kernel may read only
        a prefix of the row, so a permutation silently changes which KV the
        model attends to."""
        for dcp_size in DCP_SIZES:
            topk = _topk_rows(6, 32, 512, seed=23)
            for rank in range(dcp_size):
                with self.subTest(dcp_size=dcp_size, rank=rank):
                    out = _remap(topk, dcp_size, rank)
                    for r, row in enumerate(topk):
                        expected = [
                            int(p) // dcp_size
                            for p in row.tolist()
                            if int(p) % dcp_size == rank
                        ]
                        self.assertEqual(_valid(out[r]), expected)

    def test_padding_never_becomes_a_real_index(self):
        """The trap this guard exists for.

        The indexer pads short rows with -1. Torch's modulo follows Python, so
        ``-1 % dcp_size`` is ``dcp_size - 1`` -- meaning on the *highest* rank
        every padding entry looks owned, and ``-1 // dcp_size`` is -1 rather
        than something obviously wrong. Without the ``>= 0`` guard this test
        fails on exactly one rank out of dcp_size, which is the sort of thing a
        4-rank smoke test finds and a 2-rank one does not.
        """
        for dcp_size in DCP_SIZES:
            top_rank = dcp_size - 1
            self.assertEqual(PAD % dcp_size, top_rank, "premise of this test")
            topk = torch.full((3, 12), PAD, dtype=torch.int32)
            topk[0, 0] = top_rank  # one genuinely owned position, to prove the
            topk[1, :3] = torch.tensor(  # test is not vacuous
                [top_rank, top_rank + dcp_size, top_rank + 2 * dcp_size],
                dtype=torch.int32,
            )
            with self.subTest(dcp_size=dcp_size):
                out = _remap(topk, dcp_size, top_rank)
                self.assertEqual(_valid(out[0]), [0])
                self.assertEqual(_valid(out[1]), [0, 1, 2])
                self.assertEqual(_valid(out[2]), [])
                self.assertEqual(int((out == PAD).sum()), 36 - 4)

    def test_pads_are_a_suffix_of_every_row(self):
        # Relied on by _valid() above, and by any kernel that stops at the first
        # invalid entry rather than scanning the whole row.
        for dcp_size in DCP_SIZES:
            topk = _topk_rows(8, 20, 300, seed=31)
            for rank in range(dcp_size):
                with self.subTest(dcp_size=dcp_size, rank=rank):
                    out = _remap(topk, dcp_size, rank)
                    for row in out.tolist():
                        seen_pad = False
                        for v in row:
                            if v == PAD:
                                seen_pad = True
                            else:
                                self.assertFalse(
                                    seen_pad, f"real index after a pad in {row}"
                                )

    def test_local_indices_stay_inside_this_rank_shard(self):
        """An index past the end of the shard is an out-of-bounds read of the
        latent KV pool, not a wrong answer -- worth its own assertion because
        P2 sized that pool at exactly ``max_total // dcp_size``."""
        for dcp_size in DCP_SIZES:
            for seq_len in (dcp_size, 97, 1024):
                # Ceiling, because rank r owns position r first: with 97
                # positions over 4 ranks, rank 0 holds 25 and rank 3 holds 24.
                shard = (seq_len + dcp_size - 1) // dcp_size
                topk = torch.arange(seq_len, dtype=torch.int32).reshape(1, -1)
                for rank in range(dcp_size):
                    with self.subTest(dcp_size=dcp_size, seq_len=seq_len, rank=rank):
                        out = _remap(topk, dcp_size, rank)
                        real = out[out != PAD]
                        if real.numel():
                            self.assertGreaterEqual(int(real.min()), 0)
                            self.assertLess(int(real.max()), shard)

    def test_a_row_this_rank_owns_nothing_of_is_all_padding(self):
        # Reachable whenever the top-k for a query happens to miss this shard,
        # and at the start of decode when seq_len < dcp_size there are ranks
        # that own nothing at all. The row must still come back the right shape.
        for dcp_size in DCP_SIZES:
            rank = 0
            topk = torch.full((2, 8), rank + 1, dtype=torch.int32)
            with self.subTest(dcp_size=dcp_size):
                out = _remap(topk, dcp_size, rank)
                if dcp_size > 1:
                    self.assertEqual(out.shape, topk.shape)
                    self.assertTrue(bool((out == PAD).all()))

    def test_it_works_on_a_three_dimensional_top_k(self):
        # The indexer returns [T, K] today (dsa_npu_indexer.py:298) but the
        # speculative path carries a draft dimension; the remap is written
        # against the last axis so both shapes work, and that should not
        # silently regress.
        topk = _topk_rows(12, 16, 256, seed=41).reshape(3, 4, 16)
        dcp_size, rank = 4, 2
        out = _remap(topk, dcp_size, rank)
        self.assertEqual(out.shape, topk.shape)
        flat_out = _remap(topk.reshape(-1, 16), dcp_size, rank)
        self.assertTrue(torch.equal(out.reshape(-1, 16), flat_out))


if __name__ == "__main__":
    unittest.main()
