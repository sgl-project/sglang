"""CPU unit test for C1's packed rank-major read of the DCP extend gather.

Pins ``plan_dcp_packed_read`` / ``packed_row_of`` / ``remap_topk_to_packed``
(``layers/dcp/layout.py``). The extend gather currently permutes the all-gather's
rank-major output back into position order, which costs a context-sized
``index_select`` -- 216 ms a forward -- and the ~1.04 GiB buffer it writes into.
Neither is necessary: the sparse operator reaches its KV through
``sparse_indices``, so it does not care what order the rows sit in. Remap the
indices instead and read the collective's own output.

The failure this guards against is silent. A wrong remap does not crash; it
points the operator at a real row that belongs to a different position, and the
model answers fluently while ignoring the prompt. So the test never checks the
formula against itself. It BUILDS the buffer the collective would produce --
each rank's owned positions, padded, concatenated rank-major, then the chunk's
own KV -- and requires the remap to land on the value that position actually
holds.

Usage:
    python -m pytest test_dcp_packed_read.py -v
    python test_dcp_packed_read.py
"""

import itertools
import unittest

import torch

from sglang.srt.layers.dcp.layout import (
    packed_row_of,
    plan_dcp_packed_read,
    remap_topk_to_packed,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _build_packed(prefix_len, extend_len, dcp_size):
    """What the gather leaves in the buffer, as labels rather than KV rows.

    ``("prefix", p)`` is prefix position p, ``("extend", j)`` the chunk's own
    row j, ``("pad", ...)`` a row a rank sent only to keep every send the same
    length. Mirrors ``_dcp_gather_extend_kv_npu``: all-gather concatenates the
    sends rank-major, then this chunk's KV is appended.
    """
    send_rows = -(-prefix_len // dcp_size)
    packed = []
    for rank in range(dcp_size):
        own = [("prefix", p) for p in range(rank, prefix_len, dcp_size)]
        own += [("pad", rank, j) for j in range(len(own), send_rows)]
        packed.extend(own)
    gathered_rows = send_rows * dcp_size
    assert len(packed) == gathered_rows
    packed.extend(("extend", j) for j in range(extend_len))
    return packed


class TestDcpPackedRead(CustomTestCase):
    def _check_shape(self, prefix_len, extend_len, dcp_size):
        plan = plan_dcp_packed_read(prefix_len, extend_len, dcp_size)
        packed = _build_packed(prefix_len, extend_len, dcp_size)
        self.assertEqual(len(packed), plan.rows)

        rows = []
        for pos in range(prefix_len + extend_len):
            row = packed_row_of(pos, plan)
            rows.append(row)
            want = ("prefix", pos) if pos < prefix_len else ("extend", pos - prefix_len)
            self.assertEqual(
                packed[row],
                want,
                f"P={prefix_len} E={extend_len} C={dcp_size}: position {pos} "
                f"maps to row {row}, which holds {packed[row]}",
            )

        # Injective, in range, and everything it skips is padding.
        self.assertEqual(len(set(rows)), len(rows), "two positions share a row")
        unread = set(range(plan.rows)) - set(rows)
        self.assertTrue(all(packed[r][0] == "pad" for r in unread))
        self.assertEqual(len(unread), plan.gathered_rows - prefix_len)
        self.assertLess(len(unread), dcp_size)

    def test_served_shape(self):
        """The 972k-context tail this exists for. A served prefix is page-aligned."""
        plan = plan_dcp_packed_read(958464, 13855, 16)
        self.assertEqual(plan.send_rows, 59904)
        self.assertEqual(plan.gathered_rows, 958464)
        self.assertEqual(plan.rows, 972319)
        # 2048-token allocator pages under DCP mean no padding rows at all.
        self.assertEqual(plan.gathered_rows, 958464)
        self._check_shape(958464, 64, 16)

    def test_small_shapes_exhaustively(self):
        for dcp_size in (2, 4, 8):
            for prefix_len in range(0, 3 * dcp_size + 1):
                for extend_len in (0, 1, dcp_size, dcp_size + 1):
                    self._check_shape(prefix_len, extend_len, dcp_size)

    def test_ragged_prefixes(self):
        """P % C != 0 leaves padding rows; they must never be addressed."""
        for dcp_size in (4, 16):
            for prefix_len in (997, 2049, 16383, 49151):
                self._check_shape(prefix_len, 137, dcp_size)

    def test_remap_matches_the_reference_elementwise(self):
        for prefix_len, extend_len, dcp_size in itertools.product(
            (0, 1, 31, 997, 16384), (0, 7, 137), (2, 4, 16)
        ):
            plan = plan_dcp_packed_read(prefix_len, extend_len, dcp_size)
            probe = torch.arange(-3, prefix_len + extend_len, dtype=torch.int64)
            want = torch.tensor(
                [packed_row_of(int(v), plan) for v in probe], dtype=torch.int64
            )
            torch.testing.assert_close(remap_topk_to_packed(probe, plan), want)

    def test_remap_preserves_the_invalid_sentinel(self):
        """-1 is "no more entries" to the operator; a real row would add keys."""
        plan = plan_dcp_packed_read(16384, 128, 16)
        topk = torch.tensor([[0, 5, -1, -1], [16383, 16384, 16511, -1]])
        out = remap_topk_to_packed(topk, plan)
        self.assertEqual(out[0, 2].item(), -1)
        self.assertEqual(out[0, 3].item(), -1)
        self.assertEqual(out[1, 3].item(), -1)
        self.assertEqual(out[1, 1].item(), plan.gathered_rows)
        self.assertEqual(out.shape, topk.shape)
        self.assertEqual(out.dtype, topk.dtype)

    def test_remap_keeps_shape_and_dtype_for_int32(self):
        plan = plan_dcp_packed_read(4096, 64, 8)
        topk = torch.randint(0, 4160, (7, 2048), dtype=torch.int32)
        out = remap_topk_to_packed(topk, plan)
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.shape, topk.shape)
        self.assertTrue(int(out.max()) < plan.rows)
        self.assertTrue(int(out.min()) >= 0)


if __name__ == "__main__":
    unittest.main()
