"""CPU unit test for the DCP packed read (C1 + D2v2).

Pins the packed rank-major buffer layout against its scalar reference:

1. ``remap_topk_to_packed`` (the tensor form the indexer remap runs) must be
   element-identical to ``packed_row_of`` (the per-position reference) for
   every prefix length / dcp size / chunk count combination probed, including
   the -1 pad sentinels and the extend tail.
2. The D2v2 chunked sub-all-gather slice math lands every local row in the
   same buffer offset the monolithic (chunks == 1) rank-major gather would.
3. ``plan_dcp_packed_read`` sizes: gathered_rows == send_rows * dcp_size,
   rows == gathered_rows + extend_len.

Usage:
    python -m pytest test_dcp_packed_read.py -v
    python test_dcp_packed_read.py
"""

import unittest

import torch

from sglang.srt.layers.dcp.layout import (
    DcpPackedReadPlan,
    packed_row_of,
    plan_dcp_packed_read,
    remap_topk_to_packed,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestPackedRowOf(unittest.TestCase):
    def test_scalar_matches_layout_per_position(self):
        for prefix_len, extend_len, dcp_size in [
            (64, 16, 1),
            (65, 16, 2),
            (64, 16, 4),
            (67, 5, 8),
            (128, 1, 8),
            (1, 1, 4),
        ]:
            plan = plan_dcp_packed_read(prefix_len, extend_len, dcp_size)
            for chunks in {1, plan.chunks, 3}:
                if chunks != 1 and chunks > plan.send_rows:
                    chunks = max(1, plan.send_rows)
                mine = DcpPackedReadPlan(
                    prefix_len=plan.prefix_len,
                    extend_len=plan.extend_len,
                    dcp_size=plan.dcp_size,
                    send_rows=plan.send_rows,
                    gathered_rows=plan.gathered_rows,
                    rows=plan.rows,
                    chunks=chunks,
                )
                for pos in list(range(-2, prefix_len + extend_len + 2)) + [-1]:
                    self.assertEqual(
                        packed_row_of(pos, mine),
                        self._reference_row_of(pos, mine),
                        f"prefix={prefix_len} ext={extend_len} dcp={dcp_size} "
                        f"chunks={chunks} pos={pos}",
                    )

    @staticmethod
    def _reference_row_of(pos, plan):
        """Independent brute-force reference: replay the gather layout.

        Rank r sends local rows [0, local_r) where local_r = prefix//C + (r <
        prefix%C), zero-padded to send_rows; the rank-major gather lands rank
        r's local row i at r*send_rows + i. Chunked layouts fill chunk-major
        blocks instead. Extend rows follow the gathered prefix.
        """
        if pos < 0:
            return pos
        if pos >= plan.prefix_len:
            return plan.gathered_rows + (pos - plan.prefix_len)
        c = plan.dcp_size
        r = pos % c
        i = pos // c
        if plan.chunks == 1:
            return r * plan.send_rows + i
        q, rem = divmod(plan.send_rows, plan.chunks)
        n = plan.chunks
        k = min(i // q, n - 1) if q > 0 else n - 1
        j = i - k * q
        sk = q + rem if k == n - 1 else q
        return c * (k * q) + r * sk + j


class TestRemapTopkToPacked(unittest.TestCase):
    def test_tensor_matches_scalar(self):
        gen = torch.Generator().manual_seed(0)
        for prefix_len, extend_len, dcp_size in [(64, 16, 2), (67, 5, 8), (300, 40, 4)]:
            plan = plan_dcp_packed_read(prefix_len, extend_len, dcp_size)
            for chunks in {1, plan.chunks}:
                mine = plan._replace(chunks=chunks)
                tokens, k = 7, 5
                # Random positions in [-1, prefix+extend): -1 pads stay -1.
                pos = torch.randint(
                    -1, prefix_len + extend_len, (tokens, k), generator=gen
                )
                remapped = remap_topk_to_packed(pos, mine)
                ref = torch.tensor(
                    [[packed_row_of(int(p), mine) for p in row] for row in pos.tolist()]
                )
                self.assertTrue(torch.equal(remapped, ref))

    def test_negative_sentinels_pass_through(self):
        plan = plan_dcp_packed_read(64, 16, 4)
        pos = torch.tensor([[-1, -1, 0, 65, 79]])
        out = remap_topk_to_packed(pos, plan)
        self.assertEqual(out[0, :2].tolist(), [-1, -1])
        self.assertEqual(int(out[0, 2]), packed_row_of(0, plan))
        self.assertEqual(int(out[0, 3]), packed_row_of(65, plan))
        self.assertEqual(int(out[0, 4]), packed_row_of(79, plan))


class TestChunkedGatherLandsLikeMonolithic(unittest.TestCase):
    def test_chunks_one_is_rank_major_identity(self):
        # The chunks == 1 sub-collective writes out[:gathered] from the same
        # send the monolithic gather would: offsets must match r*send_rows+i.
        # layout.py holds the chunk count in a module constant (default 8);
        # patch it to exercise the monolithic layout.
        import unittest.mock

        import sglang.srt.layers.dcp.layout as layout

        with unittest.mock.patch.object(layout, "_DCP_GATHER_CHUNKS", 1):
            plan = plan_dcp_packed_read(64, 16, 4)
        self.assertEqual(plan.chunks, 1)
        for r in range(plan.dcp_size):
            for i in range(plan.send_rows):
                pos = r + i * plan.dcp_size
                if pos < plan.prefix_len:
                    self.assertEqual(packed_row_of(pos, plan), r * plan.send_rows + i)


class TestPlanSizes(unittest.TestCase):
    def test_sizes(self):
        plan = plan_dcp_packed_read(100, 10, 4)
        self.assertEqual(plan.send_rows, 25)
        self.assertEqual(plan.gathered_rows, 100)
        self.assertEqual(plan.rows, 110)
        plan = plan_dcp_packed_read(101, 10, 4)
        self.assertEqual(plan.send_rows, 26)
        self.assertEqual(plan.gathered_rows, 104)
        self.assertEqual(plan.rows, 114)

    def test_chunks_clamped_to_send_rows(self):
        plan = plan_dcp_packed_read(2, 1, 4)  # send_rows=1 < any chunk count
        self.assertEqual(plan.chunks, 1)


if __name__ == "__main__":
    unittest.main()
