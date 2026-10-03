"""CPU unit test for the NPU DCP extend-gather plan.

Pins ``plan_dcp_extend_gather`` (``layers/dcp/layout.py``). Under DCP each rank
holds a 1/dcp_size share of every request's prefix KV. At extend the NPU path
all-gathers those shares piece by piece -- a piece's scratch is its gathered
rows, then this chunk's own KV for the requests that end in it -- and writes
each piece into its place in the output with one ``index_select``, so no more
than one piece is ever held beside the output. The plan is the pieces and their
indices.

A wrong entry puts real KV at the wrong position, and attention reads it without
complaint -- output stays fluent and stops following the prompt. So the test
builds every rank's share from a known KV, lays each piece out the way
``all_gather_into_tensor`` does (rank 0's rows, then rank 1's, ...), fills
padding and the output with NaN, and requires the result to equal the
position-ordered KV exactly. A pad row, a neighbour's row or a row no piece
wrote cannot pass.

Usage:
    python -m pytest test_dcp_extend_gather_plan.py -v
    python test_dcp_extend_gather_plan.py
"""

import unittest

import torch

from sglang.srt.layers.dcp import layout as dcp_layout
from sglang.srt.layers.dcp.layout import (
    dcp_extend_gather_buffer,
    plan_dcp_extend_gather,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 3, 4, 16]
DIM = 3
# The SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS default the NPU path serves with.
SERVED_PIECE_ROWS = 1 << 18
# From a cut at every local row up to the served budget, where every case
# below fits in one piece.
PIECE_ROWS = [1, 5, 16, 64, 4096, SERVED_PIECE_ROWS]

# (prefix_lens, extend_lens). The first rows are what a prefix-cache hit and a
# chunked prefill produce -- page-aligned prefixes -- and the rest are the
# general case the plan still has to get right.
CASES = [
    ([0], [7]),
    ([16], [5]),
    ([2048 * 3], [1558]),
    ([16384, 32768], [16384, 1]),
    ([5], [3]),
    ([1], [1]),
    ([33, 0, 48, 7], [1, 9, 16, 2]),
    ([48, 5], [0, 3]),
    ([0, 0], [4, 4]),
]


def _combinations():
    for prefix_lens, extend_lens in CASES:
        for dcp_size in DCP_SIZES:
            for piece_rows in PIECE_ROWS:
                # Thousands of one-row pieces test nothing a few hundred do not.
                if sum(prefix_lens) <= 256 * piece_rows:
                    yield prefix_lens, extend_lens, dcp_size, piece_rows


def _gather(prefix_lens, extend_lens, dcp_size, piece_rows, seed=0):
    """Run the plan against a simulated all-gather. Returns (out, expected, plans)."""
    g = torch.Generator().manual_seed(seed)
    prefixes = [torch.randn(p, 1, DIM, generator=g) for p in prefix_lens]
    extends = [torch.randn(e, 1, DIM, generator=g) for e in extend_lens]
    plans = [
        plan_dcp_extend_gather(prefix_lens, extend_lens, dcp_size, rank, piece_rows)
        for rank in range(dcp_size)
    ]

    sends = []
    for rank, plan in enumerate(plans):
        # What the planner lists for this rank: positions rank, rank + dcp_size,
        # ... of each request, concatenated per request.
        local = [kv[rank::dcp_size] for kv in prefixes]
        send = torch.full((plan.send_rows, 1, DIM), float("nan"))
        dst = 0
        for shard, local_len, padded_len in zip(
            local, plan.local_lens, plan.padded_lens
        ):
            assert shard.shape[0] == local_len, (shard.shape[0], local_len)
            send[dst : dst + local_len] = shard
            dst += padded_len
        sends.append(send)

    own = torch.cat(extends)
    expected = torch.cat([torch.cat([p, e]) for p, e in zip(prefixes, extends)])
    out = torch.full_like(expected, float("nan"))
    for piece in plans[0].pieces:
        gathered = [send[piece.send_start : piece.send_end] for send in sends]
        scratch = torch.cat(gathered + [own[piece.extend_start : piece.extend_end]])
        out[piece.out_start : piece.out_end] = scratch.index_select(0, piece.index)
    return out, expected, plans


class TestDcpExtendGatherPlan(CustomTestCase):
    def test_the_pieces_rebuild_every_request_in_position_order(self):
        for args in _combinations():
            with self.subTest(args=args):
                out, expected, _ = _gather(*args)
                self.assertTrue(torch.equal(out, expected))

    def test_every_rank_plans_the_same_collectives(self):
        # all_gather_into_tensor needs equal sends on every rank, and every rank
        # must run the same collectives in the same order.
        for args in _combinations():
            _, _, plans = _gather(*args)
            for plan in plans[1:]:
                self.assertEqual(plan.send_rows, plans[0].send_rows)
                self.assertEqual(len(plan.pieces), len(plans[0].pieces))
                for piece, first in zip(plan.pieces, plans[0].pieces):
                    self.assertEqual(piece[:6], first[:6])
                    self.assertTrue(torch.equal(piece.index, first.index))

    def test_the_ranks_partition_each_prefix(self):
        for prefix_lens, extend_lens in CASES:
            for dcp_size in DCP_SIZES:
                plans = [
                    plan_dcp_extend_gather(
                        prefix_lens, extend_lens, dcp_size, rank, SERVED_PIECE_ROWS
                    )
                    for rank in range(dcp_size)
                ]
                for i, prefix_len in enumerate(prefix_lens):
                    self.assertEqual(
                        sum(plan.local_lens[i] for plan in plans), prefix_len
                    )

    def test_the_pieces_tile_the_sends_the_extends_and_the_output(self):
        for prefix_lens, extend_lens, dcp_size, piece_rows in _combinations():
            plan = plan_dcp_extend_gather(
                prefix_lens, extend_lens, dcp_size, 0, piece_rows
            )
            send = extend = out = 0
            for piece in plan.pieces:
                self.assertEqual(
                    (piece.send_start, piece.extend_start, piece.out_start),
                    (send, extend, out),
                )
                send, extend, out = piece.send_end, piece.extend_end, piece.out_end
                self.assertLessEqual(
                    piece.send_end - piece.send_start, max(1, piece_rows // dcp_size)
                )
                self.assertEqual(piece.index.numel(), piece.out_end - piece.out_start)
                # the scratch the gather slices per piece, which the plan sizes
                self.assertEqual(
                    piece.scratch_rows,
                    (piece.send_end - piece.send_start) * dcp_size
                    + (piece.extend_end - piece.extend_start),
                )
                self.assertGreaterEqual(int(piece.index.min()), 0)
                self.assertLess(int(piece.index.max()), piece.scratch_rows)
            self.assertEqual(
                (send, extend, out),
                (plan.send_rows, sum(extend_lens), sum(prefix_lens) + sum(extend_lens)),
            )
            self.assertEqual(
                plan.scratch_rows, max((p.scratch_rows for p in plan.pieces), default=0)
            )


class TestDcpExtendGatherBuffer(CustomTestCase):
    """Pins ``dcp_extend_gather_buffer``, the reuse the gather allocates from.

    The gather's three context-sized tensors -- the latent output, the rope-key
    output and the per-piece scratch -- used to be allocated on each of 78
    layers; they are now reserved once and sliced. The failure that reuse
    introduces is aliasing: two tensors that must differ handed the same
    storage writes one over the other, and attention then reads real KV from
    the wrong place, which this model answers fluently and off-prompt rather
    than crashing. So the keying is what these tests are about.
    """

    def setUp(self):
        super().setUp()
        dcp_layout._dcp_extend_gather_buffers.clear()
        self.addCleanup(dcp_layout._dcp_extend_gather_buffers.clear)
        # Only dtype, device and the trailing shape are read off the reference,
        # so a zero-row one is enough and costs nothing.
        self.ref = torch.empty((0, 4), dtype=torch.float32)

    def test_two_names_with_identical_shapes_get_different_storage(self):
        # THE aliasing case. The latent and the rope key are asked for with the
        # same row count on every layer; if the name were not in the key they
        # would share one buffer and the second gather would overwrite the
        # first. Held at once, so comparing addresses is sound.
        latent = dcp_extend_gather_buffer("latent", self.ref, 8)
        rope = dcp_extend_gather_buffer("rope", self.ref, 8)
        self.assertNotEqual(latent.data_ptr(), rope.data_ptr())

    def test_the_row_shape_is_part_of_the_key(self):
        wide = dcp_extend_gather_buffer("x", torch.empty((0, 4)), 8)
        narrow = dcp_extend_gather_buffer("x", torch.empty((0, 2)), 8)
        self.assertNotEqual(wide.data_ptr(), narrow.data_ptr())
        self.assertEqual(tuple(wide.shape), (8, 4))
        self.assertEqual(tuple(narrow.shape), (8, 2))

    def test_the_same_request_twice_reuses_one_allocation(self):
        # What makes this worth doing at all: layer 2..78 must not allocate.
        first = dcp_extend_gather_buffer("latent", self.ref, 1024)
        second = dcp_extend_gather_buffer("latent", self.ref, 1024)
        self.assertEqual(first.data_ptr(), second.data_ptr())

    def test_the_result_is_a_view_and_writes_reach_the_next_caller(self):
        # The gather writes through the returned tensor (index_select(out=...)),
        # so a copy would silently drop every layer's result.
        view = dcp_extend_gather_buffer("latent", self.ref, 8)
        view.fill_(3.5)
        self.assertTrue(
            torch.equal(
                dcp_extend_gather_buffer("latent", self.ref, 8),
                torch.full((8, 4), 3.5),
            )
        )


if __name__ == "__main__":
    unittest.main()
