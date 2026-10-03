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
"""CPU unit test for the NPU DCP extend-gather PREFETCH.

[Test Category] Correctness
[Test Target] hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py::
              _dcp_gather_extend_kv_npu, _issue_dcp_gather_prefetch

The prefetch splits the per-layer gather in two: the all-gathers, which read
only prefix rows earlier forwards wrote, and the rest, which needs this layer's
own KV. The first half runs for layer L+1 while layer L computes, on a side
stream, into one of two scratch slots picked by layer parity.

No device here, so this models the gather in pure Python over the real plan and
pins the two properties that do not need one:

1. Splitting the loop changes nothing -- same position-ordered output, for one
   key and for two. A prefetch carrying only the first key would leave the
   second holding the previous layer's rows: fluent, wrong, silent.
2. The handshake is what makes it safe. Negative controls remove each wait in
   turn to prove the assertions can fire. The release wait buys correctness
   (one slot is correct, just serialised); the second slot buys speed.

Values carry the layer that wrote them, so a stale read is a mismatch.
"""

import unittest

from sglang.srt.layers.dcp.layout import plan_dcp_extend_gather
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 4, 8, 16]
# The prefetch requires a single-piece plan, which is what
# SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS <= 0 produces. The inline path has no
# such restriction and the DEFAULT budget cuts a long prefix into several
# pieces, so it is tested at both -- pieces share one scratch, and gathering
# them all before consuming any would put real KV at the wrong positions.
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


class StaleRead(AssertionError):
    """A stream read or wrote a scratch slot the other was still using."""


def run_forward(
    prefix_lens,
    extend_lens,
    dcp_size,
    prefetch,
    n_keys=1,
    slots_n=2,
    release_wait=True,
    ready_wait=True,
    prefetch_keys=None,
    piece_rows=ONE_PIECE,
    hoist=False,
):
    """Simulate one forward over ``LAYERS`` layers; return [[per key] per layer].

    ``prefetch_keys`` limits which keys the side stream gathers, so a prefetch
    that drops one can be reproduced deliberately.
    """
    plans = [
        plan_dcp_extend_gather(prefix_lens, extend_lens, dcp_size, rank, piece_rows)
        for rank in range(dcp_size)
    ]
    plan = plans[0]
    if prefetch:
        assert len(plan.pieces) <= 1, "the prefetch only runs on a single-piece plan"
    total = plan.pieces[-1].out_end if plan.pieces else 0
    if prefetch_keys is None:
        prefetch_keys = range(n_keys)

    # [key][slot], mirroring _dcp_extend_gather_scratches: latent_scratch and
    # rope_scratch, each with a "_b" twin for the odd layers.
    slots = [
        [[None] * plan.scratch_rows for _ in range(slots_n)] for _ in range(n_keys)
    ]
    unreleased = [[False] * slots_n for _ in range(n_keys)]
    filled = [[False] * slots_n for _ in range(n_keys)]
    inflight = [[False] * slots_n for _ in range(n_keys)]

    def gather_piece(piece, slot, layer, on_side, keys):
        for key in keys:
            if on_side and unreleased[key][slot] and not release_wait:
                raise StaleRead("side stream refilled a slot the main stream had read")
            unreleased[key][slot] = False
            sends = [
                padded_send(
                    local_shard(prefix_lens, dcp_size, rk, layer, key), plans[rk]
                )
                for rk in range(dcp_size)
            ]
            send_len = piece.send_end - piece.send_start
            for rk in range(dcp_size):
                for j in range(send_len):
                    slots[key][slot][rk * send_len + j] = sends[rk][
                        piece.send_start + j
                    ]
            if on_side:
                inflight[key][slot] = True
            else:
                filled[key][slot] = True

    def wait_ready(slot, keys):
        for key in keys:
            if inflight[key][slot]:
                inflight[key][slot] = False
                filled[key][slot] = True

    def consume_piece(piece, slot, layer, outs):
        for key in range(n_keys):
            if not filled[key][slot]:
                raise StaleRead(
                    "main stream read a slot the side stream had not filled"
                )
            extend_rows = [
                (layer, key, r, "e", i)
                for r, e in enumerate(extend_lens)
                for i in range(e)
            ]
            gathered = (piece.send_end - piece.send_start) * dcp_size
            scratch = slots[key][slot]
            for k, j in enumerate(range(piece.extend_start, piece.extend_end)):
                scratch[gathered + k] = extend_rows[j]
            outs[key][piece.out_start : piece.out_end] = [
                scratch[i] for i in piece.index
            ]

    def layer_pass(slot, layer, prefetched):
        """One layer. Pieces share a scratch, so gather and consume interleave."""
        if prefetched:
            if ready_wait:
                wait_ready(slot, range(n_keys))
        outs = [[None] * total for _ in range(n_keys)]
        if hoist:
            # The bug this guards: every gather before any index_select.
            for piece in plan.pieces:
                gather_piece(piece, slot, layer, False, range(n_keys))
            for piece in plan.pieces:
                consume_piece(piece, slot, layer, outs)
        else:
            for piece in plan.pieces:
                if not prefetched:
                    gather_piece(piece, slot, layer, False, range(n_keys))
                consume_piece(piece, slot, layer, outs)
        for key in range(n_keys):
            filled[key][slot] = False
            unreleased[key][slot] = True
        return outs

    results, pending = [], set()
    for layer in range(LAYERS):
        slot = (layer % slots_n) if prefetch else 0
        results.append(layer_pass(slot, layer, prefetch and layer in pending))
        if prefetch and layer + 1 < LAYERS:
            for piece in plan.pieces:
                gather_piece(
                    piece, (layer + 1) % slots_n, layer + 1, True, prefetch_keys
                )
            pending.add(layer + 1)
    return results


class TestDcpExtendGatherPrefetch(CustomTestCase):
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
        self._check(prefetch=False, n_keys=2)

    def test_the_inline_path_is_right_when_the_plan_has_several_pieces(self):
        # The default SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS cuts a long
        # prefix into several pieces that share one scratch buffer, so the
        # gather and the index_select have to interleave per piece.
        self._check(
            prefetch=False, n_keys=1, piece_rows=SMALL_PIECES, cases=MULTI_PIECE_CASES
        )
        self._check(
            prefetch=False, n_keys=2, piece_rows=SMALL_PIECES, cases=MULTI_PIECE_CASES
        )

    def test_prefetching_a_layer_ahead_changes_nothing(self):
        self._check(prefetch=True, n_keys=1)

    def test_a_bf16_cache_carries_both_keys_through_the_prefetch(self):
        self._check(prefetch=True, n_keys=2)

    def test_hoisting_every_gather_ahead_of_every_consume_corrupts_pieces(self):
        # The control for the interleaving above. Pieces share one scratch, so
        # gathering them all first means piece n+1 overwrites piece n before it
        # is read, and the output carries real KV at the wrong positions --
        # fluent and wrong, with nothing raised. A single-piece plan is exempt,
        # which is why the prefetch is allowed to hoist.
        case = ([129, 7, 64], [9, 2, 6])
        got = run_forward(*case, 8, False, piece_rows=SMALL_PIECES, hoist=True)
        self.assertNotEqual(got[0][0], reference(*case, 0, 0))
        # ... and a single-piece plan is exempt, which is why the prefetch may.
        same = run_forward(*case, 8, False, hoist=True)
        self.assertEqual(same[0][0], reference(*case, 0, 0))

    def test_dropping_a_key_from_the_prefetch_is_caught(self):
        # The FP8-only gate this replaced would have left the rope half holding
        # the previous layer's rows.
        with self.assertRaises(StaleRead):
            run_forward([2048, 1024], [3, 5], 8, True, n_keys=2, prefetch_keys=[0])

    def test_the_main_stream_must_wait_the_ready_event(self):
        with self.assertRaises(StaleRead):
            run_forward([2048, 1024], [3, 5], 8, True, ready_wait=False)

    def test_the_side_stream_must_wait_the_release_event(self):
        with self.assertRaises(StaleRead):
            run_forward([2048, 1024], [3, 5], 8, True, release_wait=False)

    def test_one_slot_is_correct_but_serialised(self):
        # The release wait is what makes the prefetch correct; the second
        # buffer is what lets it overlap. With one slot and the wait kept, the
        # side stream waits for the read it is about to clobber -- safe, and
        # pointless.
        got = run_forward([2048, 1024], [3, 5], 8, True, n_keys=2, slots_n=1)
        for layer, per_key in enumerate(got):
            for key, out in enumerate(per_key):
                self.assertEqual(out, reference([2048, 1024], [3, 5], layer, key))


if __name__ == "__main__":
    unittest.main()
