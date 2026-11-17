# Copyright 2023-2026 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CPU unit test for the NPU DCP extend KV prefetch (layer-ahead, two slots).

[Test Category] Correctness
[Test Target] hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py::
              _prefetch_dcp_extend_kv, maybe_prefetch_dcp_extend_kv_npu,
              _dcp_gather_extend_kv_npu (prefetch branch)

The prefetch moves the extend KV gather one layer ahead: while layer L
computes, the side stream all-gathers layer L+1's prefix KV into one of two
per-parity slot buffers. Unlike the inline path -- and unlike the
gather-prefetch its sibling test pins, which needs a single-piece plan -- this
prefetch may hoist the gathers of ALL pieces ahead of ALL consumes, because
the slot buffer gives each piece its own region: the piece's rank-major prefix
run with its own-row tail after it. The consumer then only appends the own
rows and index_selects each piece into place.

No device here, so this models the slot buffers, the entry list and the
handshake in pure Python over the real plan, and pins what does not need one:

1. Splitting the loop changes nothing -- single- and multi-piece plans, one
   key or two (latent and rope). The hoisting is only legal because the
   per-piece regions do not alias; a shared scratch corrupts under it.
2. The handshake is what makes it safe. Negative controls remove the wait or
   drop a key to prove the corruption is visible and wrong.
3. Two in flight never share a slot. The cap is two and the hook only issues
   for the immediately next layer, so the pair is always consecutive layers
   -- opposite parity, different slot. A same-parity issue is refused loudly:
   in flight it is two collectives writing one buffer, a race whose damage a
   sequential model cannot show -- which is exactly why production raises
   instead of waiting to be proven wrong.

Values carry the layer that wrote them, so a stale read is a mismatch.
"""

import unittest

from sglang.srt.layers.dcp.layout import plan_dcp_extend_gather
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 4, 8, 16]
# ONE_PIECE is what SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS <= 0 produces.
# The served default cuts a long prefix into several pieces; the inline path
# must then interleave gather and consume per piece (they share one scratch),
# while the prefetch hoists them all -- legal only because its slot buffer
# lays the pieces out end to end.
ONE_PIECE = 1 << 62
SMALL_PIECES = 64
LAYERS = 8

# Short prefixes only for the multi-piece runs: the model is O(rows x ranks)
# per piece, and a 32k prefix at piece_rows 64 would be hundreds of pieces.
MULTI_PIECE_CASES = [
    ([129], [4]),
    ([5, 9], [3, 5]),
    ([129, 7, 64], [9, 2, 6]),
    ([1, 0, 33], [1, 1, 1]),
]

CASES = [
    ([0], [1]),
    ([1], [4]),
    ([7], [17]),
    ([16], [1]),
    ([129], [4]),
    ([0, 0], [1, 1]),
    ([5, 9], [3, 5]),
    ([16384, 32768], [1, 1]),
    ([129, 7, 64], [9, 2, 6]),
    ([1, 0, 33], [1, 1, 1]),
]


def reference(prefix_lens, extend_lens, layer, key):
    """Position-ordered output: each request's prefix, then its extend rows."""
    out = []
    for r, (p, e) in enumerate(zip(prefix_lens, extend_lens)):
        out += [(layer, key, r, "p", i) for i in range(p)]
        out += [(layer, key, r, "e", i) for i in range(e)]
    return out


def local_shard(prefix_lens, dcp_size, rank, layer, key):
    """What ``get_mla_kv_buffer`` hands one rank: its prefix rows, per request."""
    rows = []
    for r, p in enumerate(prefix_lens):
        rows += [(layer, key, r, "p", i) for i in range(p) if i % dcp_size == rank]
    return rows


def padded_send(shard_rows, plan):
    send = [None] * plan.send_rows
    src = dst = 0
    for local_len, padded_len in zip(plan.local_lens, plan.padded_lens):
        send[dst : dst + local_len] = shard_rows[src : src + local_len]
        src += local_len
        dst += padded_len
    return send


def piece_layout(plan, dcp_size):
    """(offsets, rows, gathered, total) per piece -- _dcp_prefetch_piece_layout."""
    offsets, rows_list, gathered_list, total = [], [], [], 0
    for piece in plan.pieces:
        gathered = (piece.send_end - piece.send_start) * dcp_size
        offsets.append(total)
        rows_list.append(piece.scratch_rows)
        gathered_list.append(gathered)
        total += piece.scratch_rows
    return offsets, rows_list, gathered_list, total


class StaleRead(AssertionError):
    """A read of a slot the completion event had not released."""


class SlotCollision(AssertionError):
    """An issue into a slot an in-flight entry still owns."""


def run_forward(
    prefix_lens,
    extend_lens,
    dcp_size,
    prefetch,
    n_keys=1,
    ready_wait=True,
    slot_guard=True,
    shared_regions=False,
    swap_tail=False,
    drop_key=None,
    skip_consume=(),
    piece_rows=ONE_PIECE,
):
    """Simulate one forward over ``LAYERS`` layers; return [[per key] per layer].

    Mirrors the shipped protocol: each layer block starts with the hook that
    issues the next layer's gather (and ends with the same hook, which the
    first layer's attention needs to re-issue what it waited out), and each
    attention consumes its own entry, waits out anything else in flight and
    gathers inline, or skips the gather entirely (``skip_consume`` -- the
    anomaly injection the slot guard exists for).
    """
    plans = [
        plan_dcp_extend_gather(prefix_lens, extend_lens, dcp_size, rank, piece_rows)
        for rank in range(dcp_size)
    ]
    plan = plans[0]
    offsets, rows_list, gathered_list, total = piece_layout(plan, dcp_size)
    total_out = plan.pieces[-1].out_end if plan.pieces else 0
    if not plan.send_rows:
        # The issue is a no-op without prefix rows; every layer gathers inline.
        prefetch = False

    # [key][slot], mirroring _dcp_extend_prefetch_buffers: latent_prefetch and
    # rope_prefetch, each with an even-layer and an odd-layer twin.
    slots = [[[None] * total for _ in range(2)] for _ in range(n_keys)]
    inflight = [[False] * 2 for _ in range(n_keys)]
    entries = []

    def shard(rank, layer, key):
        send = padded_send(
            local_shard(prefix_lens, dcp_size, rank, layer, key), plans[rank]
        )
        return send

    def own_rows(layer, key, piece):
        return [
            (layer, key, r, "e", i) for r, e in enumerate(extend_lens) for i in range(e)
        ][piece.extend_start : piece.extend_end]

    def issue(layer):
        slot = layer % 2
        for key in range(n_keys):
            if key == drop_key:
                # The negative control: a prefetch that skips one key.
                continue
            buf = slots[key][slot]
            for p_i, piece in enumerate(plan.pieces):
                gathered = gathered_list[p_i]
                if not gathered:
                    continue
                offset = 0 if shared_regions else offsets[p_i]
                send_len = piece.send_end - piece.send_start
                for rank in range(dcp_size):
                    send = shard(rank, layer, key)
                    for j in range(send_len):
                        buf[offset + rank * send_len + j] = send[piece.send_start + j]
            inflight[key][slot] = True

    def maybe_issue(layer):
        # maybe_prefetch_dcp_extend_kv_npu: cap two, no duplicate, and the
        # in-flight entry must own the other slot.
        if len(entries) >= 2 or layer in entries:
            return
        if slot_guard and any(e % 2 == layer % 2 for e in entries):
            raise SlotCollision(
                f"layer {layer} shares its slot with in-flight {entries}"
            )
        entries.append(layer)
        issue(layer)

    def consume(layer, outs):
        slot = layer % 2
        for key in range(n_keys):
            if inflight[key][slot] and not ready_wait:
                raise StaleRead("consume read a slot its event had not released")
            inflight[key][slot] = False
        for p_i, piece in enumerate(plan.pieces):
            rows = rows_list[p_i]
            if rows <= 0:
                continue
            gathered = gathered_list[p_i]
            for key in range(n_keys):
                own = own_rows(layer, key, piece)
                region = slots[key][slot][offsets[p_i] : offsets[p_i] + rows]
                if swap_tail:
                    region[: len(own)] = own
                else:
                    region[gathered:] = own
                outs[key][piece.out_start : piece.out_end] = [
                    region[i] for i in piece.index.tolist()
                ]

    def gather_inline(layer, outs):
        # The non-prefetch branch: gather and consume interleaved per piece,
        # one transient scratch per piece.
        for p_i, piece in enumerate(plan.pieces):
            gathered = gathered_list[p_i]
            rows = rows_list[p_i]
            for key in range(n_keys):
                scratch = [None] * rows
                send_len = piece.send_end - piece.send_start
                for rank in range(dcp_size):
                    send = shard(rank, layer, key)
                    for j in range(send_len):
                        scratch[rank * send_len + j] = send[piece.send_start + j]
                scratch[gathered:] = own_rows(layer, key, piece)
                outs[key][piece.out_start : piece.out_end] = [
                    scratch[i] for i in piece.index.tolist()
                ]

    def attention(layer):
        outs = [[None] * total_out for _ in range(n_keys)]
        if layer in skip_consume:
            return outs
        if prefetch and layer in entries:
            entries.remove(layer)
            consume(layer, outs)
        else:
            # Entries with no match for this layer -- the first layer of the
            # batch, or one whose consumer never ran: nothing may stay in
            # flight on the shared group when the inline collective issues.
            for e in entries:
                for key in range(n_keys):
                    inflight[key][e % 2] = False
            entries.clear()
            gather_inline(layer, outs)
        return outs

    results = []
    for layer in range(LAYERS):
        if prefetch and layer + 1 < LAYERS:
            maybe_issue(layer + 1)
        results.append(attention(layer))
        if prefetch and layer + 1 < LAYERS:
            # The block-end hook re-issues what the first layer waited out.
            maybe_issue(layer + 1)
    return results


class TestDcpExtendKvPrefetch(CustomTestCase):
    def _check(self, prefetch, n_keys, cases=None, **kw):
        for dcp_size in DCP_SIZES:
            for prefix_lens, extend_lens in cases or CASES:
                got = run_forward(
                    prefix_lens, extend_lens, dcp_size, prefetch, n_keys=n_keys, **kw
                )
                for layer, per_key in enumerate(got):
                    for key, out in enumerate(per_key):
                        self.assertEqual(
                            out,
                            reference(prefix_lens, extend_lens, layer, key),
                            f"dcp={dcp_size} prefix={prefix_lens} keys={n_keys} "
                            f"prefetch={prefetch} layer={layer} key={key}",
                        )

    def test_inline_gathers_rebuild_every_request_in_position_order(self):
        self._check(prefetch=False, n_keys=1)
        self._check(
            prefetch=False, n_keys=1, piece_rows=SMALL_PIECES, cases=MULTI_PIECE_CASES
        )

    def test_prefetching_a_layer_ahead_changes_nothing(self):
        # The first layer waits out and drops the next layer's too-early
        # gather, gathers itself inline, and the block-end hook re-issues it:
        # every layer after the first consumes a prefetched gather.
        self._check(prefetch=True, n_keys=1)

    def test_the_prefetch_is_right_when_the_plan_has_several_pieces(self):
        # The sibling gather-prefetch needs a single-piece plan; this one
        # hoists every piece's gather because the slot buffer reserves a
        # separate region per piece.
        self.assertTrue(
            len(
                plan_dcp_extend_gather(
                    [129, 7, 64], [9, 2, 6], 8, 0, SMALL_PIECES
                ).pieces
            )
            > 1,
            "the multi-piece premise of this test",
        )
        self._check(
            prefetch=True, n_keys=1, piece_rows=SMALL_PIECES, cases=MULTI_PIECE_CASES
        )
        self._check(
            prefetch=True, n_keys=2, piece_rows=SMALL_PIECES, cases=MULTI_PIECE_CASES
        )

    def test_a_bf16_cache_carries_both_keys_through_the_prefetch(self):
        self._check(prefetch=True, n_keys=2)

    def test_hoisting_the_gathers_needs_per_piece_regions(self):
        # The control for the test above: with every piece gathered into the
        # same offset -- the shared-scratch layout the inline path is stuck
        # with -- the hoisting lets piece n+1 overwrite piece n unread.
        case = ([129, 7, 64], [9, 2, 6])
        got = run_forward(*case, 8, True, piece_rows=SMALL_PIECES, shared_regions=True)
        self.assertNotEqual(got[1][0], reference(*case, 1, 0))
        # ... and a single-piece plan is exempt, which is why the sibling
        # gather-prefetch may hoist without per-piece regions.
        same = run_forward(*case, 8, True)
        self.assertEqual(same[1][0], reference(*case, 1, 0))

    def test_own_rows_follow_the_gathered_prefix_within_each_piece(self):
        # The ZC layout is [prefix | own-tail] per piece; own rows first
        # would overwrite gathered prefix rows and shift every index.
        case = ([129, 7, 64], [9, 2, 6])
        got = run_forward(*case, 8, True, piece_rows=SMALL_PIECES, swap_tail=True)
        self.assertNotEqual(got[1][0], reference(*case, 1, 0))

    def test_dropping_a_key_from_the_prefetch_corrupts_its_output(self):
        # Both keys share one entry: a prefetch that gathers only the latent
        # half leaves the rope output wrong with nothing raised -- fluent,
        # wrong, silent, which is what the layer tags catch.
        case = ([2048, 1024], [3, 5])
        got = run_forward(*case, 8, True, n_keys=2, drop_key=1)
        self.assertEqual(got[2][0], reference(*case, 2, 0))
        self.assertNotEqual(got[2][1], reference(*case, 2, 1))

    def test_the_consumer_must_wait_the_completion_event(self):
        with self.assertRaises(StaleRead):
            run_forward([2048, 1024], [3, 5], 8, True, ready_wait=False)

    def test_two_in_flight_never_share_a_slot(self):
        # Inject the anomaly the guard exists for: a layer whose gather never
        # runs leaves its entry in flight, and the issue two layers later
        # targets the same slot. In production that is two collectives
        # writing one buffer concurrently -- a race the guard must refuse.
        with self.assertRaises(SlotCollision):
            run_forward([2048, 1024], [3, 5], 8, True, skip_consume={2})


if __name__ == "__main__":
    unittest.main()
