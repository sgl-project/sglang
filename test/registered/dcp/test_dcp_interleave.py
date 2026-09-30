"""CPU unit test for the DCP interleave: which positions each rank holds.

[Test Category] Correctness
[Test Target] layers/dcp/layout.py::plan_dcp_extend_gather
              layers/dcp/layout.py::dcp_owner_count
              layers/dcp/layout.py::localize_dcp_indices

A rank holds every ``interleave_size``-run of positions whose block index is
its rank: ``1`` is CUDA's rule and this branch's, ``page_size`` is #37787's and
vLLM-Ascend's. Three claims:

    1. at ``interleave_size == 1`` the arithmetic reproduces the old
       expressions exactly, so the flag's off position is a true no-op;
    2. ``position -> (rank, local row)`` is a bijection with dense rows, which
       is what lets a shard be a contiguous send with no holes;
    3. the extend gather still restores position order.

(3) fails silently if it fails: a wrong index gives a correctly-shaped tensor
of the wrong rows, and attention on scrambled KV degrades output rather than
crashing. So the whole gather is simulated -- pad, all-gather, splice, index.

Usage:
    python -m pytest test_dcp_interleave.py -v
    python test_dcp_interleave.py
"""

import unittest

import torch

from sglang.srt.layers.dcp.layout import (
    dcp_owner_count,
    localize_dcp_indices,
    plan_dcp_extend_gather,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

DCP_SIZES = [2, 3, 4, 8, 16]
INTERLEAVES = [1, 2, 8, 128]
PAD_VALUE = -999


def _owner(pos, dcp_size, interleave):
    return (pos // interleave) % dcp_size


def _local_row(pos, dcp_size, interleave):
    return (pos // (interleave * dcp_size)) * interleave + pos % interleave


class TestDcpInterleaveArithmetic(CustomTestCase):
    def test_interleave_one_is_the_historical_rule(self):
        # Against the literal old expressions, not a property: "off is a
        # no-op" is what lets this land before the box can check it.
        for dcp_size in DCP_SIZES:
            for length in range(0, 400):
                self.assertEqual(
                    dcp_owner_count(length, dcp_size, 0, 1),
                    -(-length // dcp_size),  # old padded_lens
                )
                for rank in range(dcp_size):
                    self.assertEqual(
                        dcp_owner_count(length, dcp_size, rank, 1),
                        length // dcp_size + int(rank < length % dcp_size),
                    )
            for pos in range(400):
                self.assertEqual(_owner(pos, dcp_size, 1), pos % dcp_size)
                self.assertEqual(_local_row(pos, dcp_size, 1), pos // dcp_size)

    def test_every_position_lands_on_exactly_one_dense_row(self):
        # No holes: the send is sized by dcp_owner_count and filled in
        # position order, so a gap shifts every later row of that rank.
        for dcp_size in (2, 4, 16):
            for interleave in INTERLEAVES:
                for length in (0, 1, 127, 128, 1000, 2048, 4097):
                    rows = {r: [] for r in range(dcp_size)}
                    for pos in range(length):
                        rows[_owner(pos, dcp_size, interleave)].append(
                            _local_row(pos, dcp_size, interleave)
                        )
                    for rank, got in rows.items():
                        with self.subTest(
                            dcp_size=dcp_size, interleave=interleave, rank=rank
                        ):
                            self.assertEqual(
                                got,
                                list(range(len(got))),
                                "local rows must be dense, in order, from 0",
                            )
                            self.assertEqual(
                                len(got),
                                dcp_owner_count(length, dcp_size, rank, interleave),
                            )

    def test_localize_agrees_with_the_plan_arithmetic(self):
        # Device-side owner test vs host-side plan: disagree and the write
        # lands on a row the gather never reads back.
        for dcp_size in (2, 4, 16):
            for interleave in INTERLEAVES:
                pos = torch.arange(3000, dtype=torch.int64)
                for rank in range(min(dcp_size, 4)):
                    local = localize_dcp_indices(pos, dcp_size, rank, interleave)
                    expected = torch.tensor(
                        [
                            _local_row(int(p), dcp_size, interleave)
                            if _owner(int(p), dcp_size, interleave) == rank
                            else -1
                            for p in pos
                        ],
                        dtype=torch.int64,
                    )
                    with self.subTest(
                        dcp_size=dcp_size, interleave=interleave, rank=rank
                    ):
                        self.assertTrue(torch.equal(local, expected))


class TestDcpExtendGatherRestoresPositionOrder(CustomTestCase):
    """Simulate the whole extend gather on CPU and check what comes out."""

    def _run(self, prefix_lens, extend_lens, dcp_size, interleave, piece_budget):
        # One distinct value per token, so a misplaced row is visible.
        def prefix_val(req, pos):
            return req * 1_000_000 + pos

        def extend_val(req, j):
            return req * 1_000_000 + prefix_lens[req] + j

        plans = [
            plan_dcp_extend_gather(
                prefix_lens, extend_lens, dcp_size, rank, piece_budget, interleave
            )
            for rank in range(dcp_size)
        ]
        # Same collectives in the same order on every rank, or it deadlocks.
        for plan in plans[1:]:
            self.assertEqual(len(plan.pieces), len(plans[0].pieces))
            self.assertEqual(plan.padded_lens, plans[0].padded_lens)
            for a, b in zip(plan.pieces, plans[0].pieces):
                self.assertEqual((a.send_start, a.send_end), (b.send_start, b.send_end))
                self.assertEqual((a.out_start, a.out_end), (b.out_start, b.out_end))
                self.assertTrue(torch.equal(a.index, b.index))

        plan = plans[0]
        padded = plan.padded_lens

        # What _pad_dcp_extend_send builds: owned rows per request, padded.
        sends = []
        for rank in range(dcp_size):
            buf = torch.full((sum(padded),), PAD_VALUE, dtype=torch.int64)
            base = 0
            for req, plen in enumerate(prefix_lens):
                for pos in range(plen):
                    if _owner(pos, dcp_size, interleave) == rank:
                        buf[base + _local_row(pos, dcp_size, interleave)] = prefix_val(
                            req, pos
                        )
                base += padded[req]
            sends.append(buf)

        own = torch.tensor(
            [extend_val(r, j) for r, e in enumerate(extend_lens) for j in range(e)],
            dtype=torch.int64,
        )
        out = torch.full((plan.pieces[-1].out_end,), PAD_VALUE, dtype=torch.int64)

        for piece in plan.pieces:
            gathered = (piece.send_end - piece.send_start) * dcp_size
            rows = gathered + piece.extend_end - piece.extend_start
            scratch = torch.full((rows,), PAD_VALUE, dtype=torch.int64)
            if gathered:
                # all_gather_into_tensor is rank-major.
                scratch[:gathered] = torch.cat(
                    [s[piece.send_start : piece.send_end] for s in sends]
                )
            scratch[gathered:] = own[piece.extend_start : piece.extend_end]
            out[piece.out_start : piece.out_end] = scratch.index_select(0, piece.index)

        # One contiguous run per request, which is what the operator needs.
        expected = []
        for req, (plen, elen) in enumerate(zip(prefix_lens, extend_lens)):
            expected += [prefix_val(req, p) for p in range(plen)]
            expected += [extend_val(req, j) for j in range(elen)]
        return out, torch.tensor(expected, dtype=torch.int64)

    def test_one_request_across_interleaves(self):
        for dcp_size in (2, 4, 16):
            for interleave in INTERLEAVES:
                for prefix in (0, 512, 2048, 4096):
                    with self.subTest(
                        dcp_size=dcp_size, interleave=interleave, prefix=prefix
                    ):
                        out, expected = self._run(
                            [prefix], [16], dcp_size, interleave, 1 << 14
                        )
                        self.assertTrue(torch.equal(out, expected))

    def test_many_requests_share_a_piece(self):
        # Small requests share a piece -- where the per-request send offset
        # in the index expression earns its keep.
        for interleave in INTERLEAVES:
            with self.subTest(interleave=interleave):
                out, expected = self._run(
                    [256, 512, 128, 1024], [4, 8, 1, 2], 4, interleave, 1 << 13
                )
                self.assertTrue(torch.equal(out, expected))

    def test_a_prefix_that_spans_several_pieces(self):
        # Small budget so one request is cut several times; cuts must fall on
        # whole blocks or a rank's run splits.
        for interleave in INTERLEAVES:
            with self.subTest(interleave=interleave):
                out, expected = self._run(
                    [8192], [32], 4, interleave, max(64, interleave * 8)
                )
                self.assertTrue(torch.equal(out, expected))

    def test_an_unaligned_prefix_still_restores(self):
        # Served prefixes are cycle-aligned; the plan pads rather than
        # assuming it, and pad rows must never reach the output.
        for interleave in (1, 8):
            for prefix in (1, 7, 129, 1023):
                with self.subTest(interleave=interleave, prefix=prefix):
                    out, expected = self._run([prefix], [3], 4, interleave, 1 << 12)
                    self.assertTrue(torch.equal(out, expected))
                    self.assertNotIn(PAD_VALUE, out.tolist())


if __name__ == "__main__":
    unittest.main()
