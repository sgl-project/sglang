"""CPU unit test for the NPU DCP extend-gather plan.

Pins ``plan_dcp_extend_gather`` (``hardware_backend/npu/dcp_extend_gather.py``). Under DCP each rank
holds a 1/dcp_size share of every request's prefix KV, in runs of
``interleave`` positions. At extend the NPU path all-gathers those shares piece
by piece -- a piece's scratch is its gathered rows, then this chunk's own KV for
the requests that end in it -- and writes each piece into its place in the
output with one ``index_select``. The plan is the pieces and their indices.

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
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt import runtime_context as rc
from sglang.srt.hardware_backend.npu import dcp_extend_gather
from sglang.srt.hardware_backend.npu.dcp_extend_gather import (
    DCP_EXTEND_QUERY_GATHER_MAX_TOKENS,
    build_dcp_extend_gather_plan,
    dcp_extend_gather_buffer,
    dcp_extend_gathers_kv,
    plan_dcp_extend_gather,
)
from sglang.srt.layers.dcp import layout as dcp_layout
from sglang.srt.layers.dcp.layout import (
    dcp_local_row,
    filter_dcp_local_chunk_kv_indices,
    localize_dcp_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 3, 4, 16]
# 1 is the round-robin rule; NPU serves at its page size.
INTERLEAVES = [1, 4, 128]
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
    ([300, 2048 + 77], [5, 6]),
]


def _owned(prefix_len, dcp_size, rank, interleave):
    """The positions of one prefix a rank holds, walked one by one."""
    return [p for p in range(prefix_len) if (p // interleave) % dcp_size == rank]


def _combinations():
    for prefix_lens, extend_lens in CASES:
        for dcp_size in DCP_SIZES:
            for interleave in INTERLEAVES:
                for piece_rows in PIECE_ROWS:
                    # Thousands of one-row pieces test nothing a few hundred do not.
                    if sum(prefix_lens) <= 256 * piece_rows:
                        yield prefix_lens, extend_lens, dcp_size, piece_rows, interleave


def _gather(prefix_lens, extend_lens, dcp_size, piece_rows, interleave, seed=0):
    """Run the plan against a simulated all-gather. Returns (out, expected, plans)."""
    g = torch.Generator().manual_seed(seed)
    prefixes = [torch.randn(p, 1, DIM, generator=g) for p in prefix_lens]
    extends = [torch.randn(e, 1, DIM, generator=g) for e in extend_lens]
    plans = [
        plan_dcp_extend_gather(
            prefix_lens, extend_lens, dcp_size, rank, piece_rows, interleave
        )
        for rank in range(dcp_size)
    ]

    sends = []
    for rank, plan in enumerate(plans):
        local = [kv[_owned(kv.shape[0], dcp_size, rank, interleave)] for kv in prefixes]
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

    def test_the_pieces_tile_the_sends_the_extends_and_the_output(self):
        for prefix_lens, extend_lens, dcp_size, piece_rows, b in _combinations():
            plan = plan_dcp_extend_gather(
                prefix_lens, extend_lens, dcp_size, 0, piece_rows, b
            )
            send = extend = out = 0
            for piece in plan.pieces:
                self.assertEqual(
                    (piece.send_start, piece.extend_start, piece.out_start),
                    (send, extend, out),
                )
                send, extend, out = piece.send_end, piece.extend_end, piece.out_end
                self.assertLessEqual(
                    piece.send_end - piece.send_start, max(b, piece_rows // dcp_size)
                )
                self.assertEqual(piece.index.numel(), piece.out_end - piece.out_start)
                self.assertGreaterEqual(int(piece.index.min()), 0)
                self.assertLess(int(piece.index.max()), piece.scratch_rows)
            self.assertEqual(
                (send, extend, out),
                (plan.send_rows, sum(extend_lens), sum(prefix_lens) + sum(extend_lens)),
            )


class TestDcpLocalChunkOffsetsAtInterleaveOne(CustomTestCase):
    """The interleaved filter's offsets, run at interleave 1, must select what
    the round-robin slice selects -- including chunks that start mid-cycle."""

    def test_matches_the_round_robin_slice(self):
        starts_cases = [[0], [5], [0, 3, 17], [1, 1, 1, 1], [33, 0, 48, 7]]
        for (prefix_lens, _), starts in zip(CASES * 2, starts_cases * 4):
            starts = (starts * len(prefix_lens))[: len(prefix_lens)]
            kv = torch.arange(sum(prefix_lens), dtype=torch.int64)
            for dcp_size in DCP_SIZES[1:]:
                for rank in range(dcp_size):
                    with rc.get_parallel().override(
                        dcp_enabled=True, dcp_size=dcp_size, dcp_rank=rank
                    ):
                        expected = filter_dcp_local_chunk_kv_indices(
                            kv, torch.tensor(starts), torch.tensor(prefix_lens)
                        )
                    offsets = dcp_layout._dcp_local_chunk_offsets(
                        starts, prefix_lens, dcp_size, rank, 1
                    )
                    self.assertEqual(
                        kv[offsets].tolist(),
                        expected.tolist(),
                        (prefix_lens, starts, dcp_size, rank),
                    )


class TestDcpLocalPrefixRows(CustomTestCase):
    """The planner's read of this rank's prefix rows -- the chunk filter, then
    ``dcp_local_row`` -- must land on the rows the pool's write put them in
    (``localize_dcp_indices``), in the order the gather plan expects. Selection
    is by position and mapping by slot, so an off-by-one reads real KV of the
    wrong position: right shapes, fluent output that stops following the prompt.

    Each rank's pool is written through the write mapping, slots come from
    widened allocator pages handed out in shuffled order, and every row read
    must carry the tag of the position it was planned for."""

    def test_every_rank_reads_back_what_the_write_stored(self):
        g = torch.Generator().manual_seed(0)
        for prefix_lens, _ in CASES:
            for dcp_size in DCP_SIZES:
                for interleave in INTERLEAVES:
                    page = dcp_size * interleave
                    pages = [-(-p // page) for p in prefix_lens]
                    order = torch.randperm(sum(pages), generator=g)
                    slots, used = [], 0
                    for prefix_len, n in zip(prefix_lens, pages):
                        pos = torch.arange(prefix_len)
                        table = order[used : used + n]
                        slots.append(table[pos // page] * page + pos % page)
                        used += n
                    slots = torch.cat(slots)
                    # Tag = offset into the concatenated prefixes, + 1 so a row
                    # that was never written (NaN) or a zero cannot pass.
                    tags = torch.arange(1, slots.numel() + 1, dtype=torch.float64)
                    for rank in range(dcp_size):
                        with self.subTest(
                            prefix_lens=prefix_lens,
                            dcp_size=dcp_size,
                            interleave=interleave,
                            rank=rank,
                        ):
                            pool = torch.full(
                                (sum(pages) * interleave,),
                                float("nan"),
                                dtype=torch.float64,
                            )
                            loc = localize_dcp_indices(
                                slots, dcp_size, rank, interleave
                            )
                            pool[loc[loc >= 0]] = tags[loc >= 0]
                            with rc.get_parallel().override(
                                dcp_enabled=dcp_size > 1,
                                dcp_size=dcp_size,
                                dcp_rank=rank,
                            ):
                                owned_slots = filter_dcp_local_chunk_kv_indices(
                                    slots,
                                    torch.zeros(len(prefix_lens), dtype=torch.int64),
                                    torch.tensor(prefix_lens),
                                    interleave_size=interleave,
                                )
                            rows = dcp_local_row(owned_slots, dcp_size, interleave)
                            owned, offset = [], 0
                            for p in prefix_lens:
                                owned += [
                                    offset + x
                                    for x in _owned(p, dcp_size, rank, interleave)
                                ]
                                offset += p
                            self.assertTrue(torch.equal(pool[rows], tags[owned]))


class TestDcpExtendGathersKv(CustomTestCase):
    """The per-forward choice between gathering the query and gathering the
    prefix KV. GLM-5.2's dims put the byte crossover at exactly prefix / 178;
    a tail past the query gather's memory cap gathers the KV regardless."""

    GLM52 = dict(num_heads=64, kv_lora_rank=512, qk_rope_head_dim=64, kv_elem_size=2)

    def test_the_crossover_is_prefix_over_178_and_a_tie_gathers_kv(self):
        self.assertTrue(dcp_extend_gathers_kv([178_000], [1000], **self.GLM52))
        self.assertFalse(dcp_extend_gathers_kv([178_000], [999], **self.GLM52))

    def test_it_sums_over_the_batch(self):
        self.assertTrue(dcp_extend_gathers_kv([89_000] * 2, [500, 500], **self.GLM52))
        self.assertFalse(dcp_extend_gathers_kv([89_000] * 2, [500, 499], **self.GLM52))

    def test_a_long_tail_gathers_kv_whatever_the_bytes(self):
        # Three long-prefix requests in one chunk favour the query by bytes, but
        # its fp32 merge does not fit beside them.
        prefix = [178 * (DCP_EXTEND_QUERY_GATHER_MAX_TOKENS + 1)] * 3
        tail = [DCP_EXTEND_QUERY_GATHER_MAX_TOKENS]
        self.assertFalse(dcp_extend_gathers_kv(prefix, tail, **self.GLM52))
        tail = [DCP_EXTEND_QUERY_GATHER_MAX_TOKENS + 1]
        self.assertTrue(dcp_extend_gathers_kv(prefix, tail, **self.GLM52))


class TestBuildDcpExtendGatherPlanDecides(CustomTestCase):
    """``build_dcp_extend_gather_plan`` is the one place a forward picks the KV
    gather; every layer only tests its result for None. Target verify and DSA
    prefill CP must get None, and a target-verify batch may carry no
    ``extend_seq_lens_cpu``, so planning one anyway crashes metadata init."""

    def _batch(self, *, extend=True, target_verify=False, extend_lens=(3,)):
        mode = SimpleNamespace(
            is_extend=lambda: extend,
            is_target_verify=lambda: target_verify,
            is_context_parallel_extend=lambda: False,
        )
        return SimpleNamespace(
            forward_mode=mode,
            attn_cp_metadata=None,
            extend_prefix_lens_cpu=[16],
            extend_seq_lens_cpu=None if extend_lens is None else list(extend_lens),
        )

    def _build(self, batch, prefill_cp=False):
        # dsa_use_prefill_cp reads server-wide CP state no CPU test sets up.
        with (
            rc.get_parallel().override(dcp_enabled=True, dcp_size=2, dcp_rank=0),
            patch.object(
                dcp_extend_gather, "dsa_use_prefill_cp", return_value=prefill_cp
            ),
        ):
            return build_dcp_extend_gather_plan(
                batch, interleave_size=4, device=torch.device("cpu")
            )

    def test_a_plain_extend_gets_a_plan(self):
        self.assertIsNotNone(self._build(self._batch()))

    def test_target_verify_gets_none_without_reading_extend_lens(self):
        batch = self._batch(target_verify=True, extend_lens=None)
        self.assertIsNone(self._build(batch))

    def test_a_forward_that_is_not_an_extend_gets_none(self):
        self.assertIsNone(self._build(self._batch(extend=False, extend_lens=None)))

    def test_dsa_prefill_cp_gets_none(self):
        self.assertIsNone(self._build(self._batch(), prefill_cp=True))


class TestDcpExtendGatherBuffer(CustomTestCase):
    """Pins ``dcp_extend_gather_buffer``, the reuse the gather allocates from.

    The failure that reuse introduces is aliasing: two tensors that must differ
    handed the same storage writes one over the other, and attention then reads
    real KV from the wrong place, which a model answers fluently and off-prompt
    rather than crashing.
    """

    def setUp(self):
        super().setUp()
        dcp_extend_gather._dcp_extend_gather_buffers.clear()
        self.addCleanup(dcp_extend_gather._dcp_extend_gather_buffers.clear)
        # Only dtype, device and the trailing shape are read off the reference.
        self.ref = torch.empty((0, 4), dtype=torch.float32)

    def test_two_names_with_identical_shapes_get_different_storage(self):
        latent = dcp_extend_gather_buffer("latent", self.ref, 8)
        rope = dcp_extend_gather_buffer("rope", self.ref, 8)
        self.assertNotEqual(latent.data_ptr(), rope.data_ptr())

    def test_the_same_request_twice_reuses_one_allocation(self):
        # What makes this worth doing at all: layers 2..78 must not allocate.
        first = dcp_extend_gather_buffer("latent", self.ref, 1024)
        second = dcp_extend_gather_buffer("latent", self.ref, 1024)
        self.assertEqual(first.data_ptr(), second.data_ptr())

    def test_the_result_is_a_view_and_writes_reach_the_next_caller(self):
        # The gather writes through the returned tensor (index_select(out=...)).
        dcp_extend_gather_buffer("latent", self.ref, 8).fill_(3.5)
        self.assertTrue(
            torch.equal(
                dcp_extend_gather_buffer("latent", self.ref, 8),
                torch.full((8, 4), 3.5),
            )
        )


if __name__ == "__main__":
    unittest.main()
