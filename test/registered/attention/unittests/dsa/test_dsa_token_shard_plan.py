"""CPU unit test for the DSA token shard plan.

Pins ``plan_dsa_token_shard`` (``layers/attention/dsa/dsa_token_shard_layout.py``),
which cuts an extend batch's tokens across the attention-TP group. The cut is by
position, not by request, so a request can straddle a slice boundary and each
rank's per-request query and key lengths have to be recomputed against its slice.

That recomputation is clamp arithmetic, and getting it wrong is silent: the
operator is handed a span that is merely different rather than invalid. So the
test does not check the clamps. It rebuilds the answer the slow way -- walk every
token, assign it to a rank, read the counts off -- and requires the two to agree
on every small batch.

The collective pair ``dsa_token_shard_redistribute_heads`` /
``dsa_token_shard_restore_tokens`` cannot run a real process group on CPU CI,
so the exchange is scripted: the test precomputes every rank's send buffer,
the fake group asserts the call under test built exactly that send, and the
recv is the deterministic all-to-all of the expected sends. That still guards
the reshape/permute math, the head order, the pad handling and the NaN
zeroing -- the places a silent-wrong-rows bug would live.

Usage:
    python -m pytest test_dsa_token_shard_plan.py -v
    python test_dsa_token_shard_plan.py
"""

import itertools
import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa.dsa_token_shard import (
    _build_dsa_token_shard_plan,
    dsa_token_shard_cumulative_query_lens,
    dsa_token_shard_key_lens,
    dsa_token_shard_redistribute_heads,
    dsa_token_shard_restore_tokens,
    dsa_token_shard_slice,
)
from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import (
    cumulative,
    plan_dsa_token_shard,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _brute(extend_seq_lens, seq_lens, tp_size, tp_rank):
    """Per-request (query_len, key_len) for one rank, walked token by token."""
    num_tokens = sum(extend_seq_lens)
    rows = -(-num_tokens // tp_size)
    lo, hi = tp_rank * rows, tp_rank * rows + rows

    query_lens, key_lens = [], []
    start = 0
    for n, seq_len in zip(extend_seq_lens, seq_lens):
        end = start + n
        # Tokens of this request inside [lo, hi), padding included -- padding
        # always falls past the last request, so it never lands here.
        q = max(0, min(end, hi) - max(start, lo))
        query_lens.append(q)
        if q == 0:
            key_lens.append(0)
        else:
            # The slice's last token of this request is at global position
            # min(end, hi) - 1, i.e. min(end, hi) - start tokens into the
            # request. It sees the prefix plus those.
            prefix = seq_len - n
            key_lens.append(max(0, prefix + (min(end, hi) - start)))
        start = end
    return query_lens, key_lens


class TestDsaTokenShardPlan(CustomTestCase):
    def _check(self, extend_seq_lens, seq_lens, tp_size, label=""):
        num_tokens = sum(extend_seq_lens)
        rows = -(-num_tokens // tp_size)
        covered = [0] * num_tokens

        for rank in range(tp_size):
            plan = plan_dsa_token_shard(extend_seq_lens, seq_lens, tp_size, rank)
            want_q, want_k = _brute(extend_seq_lens, seq_lens, tp_size, rank)

            self.assertEqual(plan.query_lens, want_q, f"{label} rank={rank} query")
            self.assertEqual(plan.key_lens, want_k, f"{label} rank={rank} key")
            self.assertEqual(plan.rows, rows, f"{label} rank={rank} rows")
            self.assertEqual(plan.num_tokens_pad, rows * tp_size, label)
            # A rank is handed `rows` query rows; describing more than that
            # would run the operator off the end of the tensor.
            self.assertLessEqual(sum(plan.query_lens), plan.rows, label)

            for pos in range(
                plan.local_start, min(plan.local_end_with_pad, num_tokens)
            ):
                covered[pos] += 1

        # Every real token computed exactly once across the group. A token
        # covered twice is wasted work; a token covered zero times is missing
        # output that nothing downstream will notice.
        self.assertTrue(all(c == 1 for c in covered), f"{label} partition")

    def test_every_small_batch_exhaustively(self):
        # 1-3 requests, 0-6 tokens each, tp 1-5. Off-by-ones live here.
        for nreq in (1, 2, 3):
            for lens in itertools.product(range(7), repeat=nreq):
                if sum(lens) == 0:
                    continue
                seqs = [n + 10 * (i + 1) for i, n in enumerate(lens)]
                for tp in (1, 2, 3, 4, 5):
                    self._check(list(lens), seqs, tp, f"{lens} tp{tp}")

    def test_mismatched_metadata_raises_rather_than_guesses(self):
        with self.assertRaises(AssertionError):
            plan_dsa_token_shard([5, 9], [105], 4, 0)
        with self.assertRaises(AssertionError):
            plan_dsa_token_shard([5], [105], 4, 4)

    def test_single_request_12119_tp16_key_lens(self):
        # Paged SFA must be handed these key lengths, not the full seq_len.
        # sparse_mode=3 right-aligns to the length it is given, so only the
        # rank holding the tail may see 118615.
        prefix, new, tp = 106496, 12119, 16
        seq = prefix + new
        plan0 = plan_dsa_token_shard([new], [seq], tp, 0)
        plan14 = plan_dsa_token_shard([new], [seq], tp, 14)
        plan15 = plan_dsa_token_shard([new], [seq], tp, 15)
        self.assertEqual(plan0.rows, 758)
        self.assertEqual(plan0.query_lens, [758])
        self.assertEqual(plan0.key_lens, [107254])
        self.assertEqual(plan14.query_lens, [758])
        self.assertEqual(plan14.key_lens, [117866])
        self.assertEqual(plan15.query_lens, [749])
        self.assertEqual(plan15.key_lens, [seq])
        self.assertNotEqual(plan0.key_lens, [seq])

        forward_batch = types.SimpleNamespace()
        self.assertEqual(
            dsa_token_shard_key_lens(
                forward_batch, plan0, torch.device("cpu")
            ).tolist(),
            [107254],
        )
        self.assertEqual(
            dsa_token_shard_key_lens(
                types.SimpleNamespace(), plan15, torch.device("cpu")
            ).tolist(),
            [seq],
        )

    def test_key_lens_are_absolute_not_a_prefix_sum(self):
        # rank 0 of tp=2, requests [5, 9] against seq [105, 214].
        # The frontier is [105, 207]. A prefix sum would be [105, 312].
        plan = plan_dsa_token_shard([5, 9], [105, 214], 2, 0)
        self.assertEqual(plan.key_lens, [105, 207])
        forward_batch = types.SimpleNamespace()
        self.assertEqual(
            dsa_token_shard_key_lens(forward_batch, plan, torch.device("cpu")).tolist(),
            [105, 207],
        )


class TestTokenShardRefusesDcp(CustomTestCase):
    """DCP's DSA path all-gathers the query across the DCP group and expects
    every token's head-sharded row; a token slice would be merged wrong."""

    def _plan(self, dcp_enabled):
        batch = types.SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[16384],
            extend_prefix_lens_cpu=[0],
            input_ids=torch.zeros(16384, dtype=torch.int64),
            attn_cp_metadata=None,
        )
        with get_parallel().override(
            attn_tp_size=16, attn_tp_rank=3, dcp_enabled=dcp_enabled
        ):
            return _build_dsa_token_shard_plan(batch)

    def test_planned_without_dcp(self):
        self.assertIsNotNone(self._plan(dcp_enabled=False))

    def test_refused_under_dcp(self):
        self.assertIsNone(self._plan(dcp_enabled=True))


class TestTokenShardRefusesPrefillCp(CustomTestCase):
    """Prefill CP hands attention its own prev/next query layout; the plan's
    cumulative query lengths would replace it and the core would slice a
    second time."""

    def _plan(self, prefill_cp):
        batch = types.SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[16384],
            extend_prefix_lens_cpu=[0],
            input_ids=torch.zeros(16384, dtype=torch.int64),
            attn_cp_metadata=object() if prefill_cp else None,
        )
        with (
            mock.patch(
                "sglang.srt.layers.attention.dsa.dsa_token_shard.is_dsa_enable_prefill_cp",
                return_value=prefill_cp,
            ),
            get_parallel().override(attn_tp_size=16, attn_tp_rank=3, dcp_enabled=False),
        ):
            return _build_dsa_token_shard_plan(batch)

    def test_planned_without_prefill_cp(self):
        self.assertIsNotNone(self._plan(prefill_cp=False))

    def test_refused_under_prefill_cp(self):
        self.assertIsNone(self._plan(prefill_cp=True))


class TestTokenShardRefusesDpMaxLenPadding(CustomTestCase):
    """Attention-DP MAX_LEN pads a rank to the widest DP rank. The plan is cut
    from real extend lengths, so a wider batch would trip the redistribute's
    width assert after the indexer had already kept only its local top-k."""

    def _plan(self, num_rows):
        batch = types.SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[1000],
            extend_prefix_lens_cpu=[0],
            input_ids=torch.zeros(num_rows, dtype=torch.int64),
            attn_cp_metadata=None,
        )
        with get_parallel().override(
            attn_tp_size=16, attn_tp_rank=3, dcp_enabled=False
        ):
            return _build_dsa_token_shard_plan(batch)

    def test_attn_tp_alignment_is_planned(self):
        # SUM_LEN pads only to a multiple of attn_tp_size: 1000 -> 1008.
        self.assertIsNotNone(self._plan(num_rows=1008))

    def test_max_len_padding_is_refused(self):
        self.assertIsNone(self._plan(num_rows=2000))


class _ScriptedAllToAllGroup:
    """Stands in for the attention-TP process group on CPU CI.

    all_to_all_single is deterministic: the recv chunk from rank s is rank s's
    send chunk for this rank. The test precomputes every rank's send, so the
    fake can assert the call under test built exactly that send and return the
    matching recv -- no second process needed.
    """

    def __init__(self, expected_sends, state):
        self._expected_sends = expected_sends
        self._state = state

    def all_to_all_single(self, recv, send):
        me = self._state["rank"]
        torch.testing.assert_close(
            send, self._expected_sends[me], rtol=0, atol=0, equal_nan=True
        )
        expected_recv = torch.stack(
            [
                self._expected_sends[sender][me]
                for sender in range(len(self._expected_sends))
            ]
        )
        assert recv.shape == expected_recv.shape, (recv.shape, expected_recv.shape)
        recv.copy_(expected_recv)


def _expected_redistribute_send(x, plan, tp, head, dim):
    """The send buffer redistribute builds: zero-pad to the planned width,
    then one chunk per rank."""
    missing = plan.num_tokens_pad - x.shape[0]
    if missing > 0:
        x = torch.cat([x, x.new_zeros((missing, *x.shape[1:]))], dim=0)
    return x.reshape(tp, plan.rows, head, dim).contiguous()


class TestTokenShardAllToAll(CustomTestCase):
    """Round-trip of the collective pair, with the exchange scripted.

    A reshape or permute slip in either function yields tensors of the right
    shapes with the wrong rows, so the tests pin the cross-rank head order
    and the exact inverse, not just "it runs".
    """

    def _run(
        self, extend_lens, seq_lens, tp, head=3, dim=8, nan_tail=False, narrow=False
    ):
        plans = [plan_dsa_token_shard(extend_lens, seq_lens, tp, r) for r in range(tp)]
        rows, pad = plans[0].rows, plans[0].num_tokens_pad
        num_tokens = plans[0].num_tokens
        full = [
            (torch.arange(pad * head * dim, dtype=torch.float32) + r * 100000)
            .reshape(pad, head, dim)
            .contiguous()
            for r in range(tp)
        ]
        if nan_tail:
            for r in range(tp):
                full[r][num_tokens:] = float("nan")
        # narrow: the batch arrives unpadded and redistribute pads it.
        inputs = [x[:num_tokens].clone() for x in full] if narrow else full
        state = {"rank": 0}
        parallel = types.SimpleNamespace(
            attn_tp_size=tp, attn_tp_rank=0, attn_tp_group=None
        )
        patch = mock.patch(
            "sglang.srt.layers.attention.dsa.dsa_token_shard.get_parallel",
            return_value=parallel,
        )
        with patch:
            expected_sends = [
                _expected_redistribute_send(inputs[r], plans[r], tp, head, dim)
                for r in range(tp)
            ]
            parallel.attn_tp_group = _ScriptedAllToAllGroup(expected_sends, state)
            ys = []
            for r in range(tp):
                state["rank"] = r
                parallel.attn_tp_rank = r
                ys.append(dsa_token_shard_redistribute_heads(inputs[r], plans[r]))
            for r in range(tp):
                for s in range(tp):
                    # Padded rows are zero in the send only when redistribute
                    # padded them (narrow); full-width inputs keep their values.
                    real = max(0, min(rows, num_tokens - r * rows))
                    got = ys[r][:real, s * head : (s + 1) * head]
                    want = full[s][r * rows : r * rows + real]
                    self.assertTrue(torch.equal(got, want))

            expected_sends = [
                ys[r].reshape(rows, tp, head, dim).permute(1, 0, 2, 3).contiguous()
                for r in range(tp)
            ]
            parallel.attn_tp_group = _ScriptedAllToAllGroup(expected_sends, state)
            for r in range(tp):
                state["rank"] = r
                parallel.attn_tp_rank = r
                num_rows = inputs[r].shape[0]
                out = dsa_token_shard_restore_tokens(ys[r], plans[r], num_rows)
                want = full[r][:num_rows].clone()
                if num_rows > num_tokens:
                    want[num_tokens:] = 0
                self.assertTrue(torch.equal(out, want))

    def test_round_trip_multi_request_tp2(self):
        self._run([5, 9], [105, 214], tp=2)

    def test_round_trip_uneven_last_slice_tp4(self):
        # 17 tokens on 4 ranks: the last slice holds 2 real tokens, the rest
        # padding -- the case the equal-slices intuition gets wrong.
        self._run([17], [117], tp=4)

    def test_round_trip_unpadded_input(self):
        # The batch can arrive at the real token count; redistribute pads it.
        # tp=4 so 14 tokens really do pad (16), unlike the tp=2 cases above.
        self._run([5, 9], [105, 214], tp=4, narrow=True)

    def test_round_trip_keeps_padded_width(self):
        # num_rows leaves padded: what the next layer's KV write indexes
        # against. Tail must come back zeroed even if the operator left NaN.
        self._run([5, 9], [105, 214], tp=2, nan_tail=True)

    def test_redistribute_rejects_overwide_input(self):
        plan = plan_dsa_token_shard([5, 9], [105, 214], 2, 0)
        x = torch.zeros(plan.num_tokens_pad + 1, 3, 8)
        state = {"rank": 0}
        parallel = types.SimpleNamespace(
            attn_tp_size=2, attn_tp_rank=0, attn_tp_group=None
        )
        with mock.patch(
            "sglang.srt.layers.attention.dsa.dsa_token_shard.get_parallel",
            return_value=parallel,
        ):
            with self.assertRaises(AssertionError):
                dsa_token_shard_redistribute_heads(x, plan)

    def test_restore_rejects_rows_outside_the_plan(self):
        plan = plan_dsa_token_shard([5, 9], [105, 214], 2, 0)
        y = torch.zeros(plan.rows, 2 * 3, 8)
        state = {"rank": 0}
        parallel = types.SimpleNamespace(
            attn_tp_size=2, attn_tp_rank=0, attn_tp_group=None
        )
        with mock.patch(
            "sglang.srt.layers.attention.dsa.dsa_token_shard.get_parallel",
            return_value=parallel,
        ):
            with self.assertRaises(AssertionError):
                dsa_token_shard_restore_tokens(y, plan, plan.num_tokens - 1)
            with self.assertRaises(AssertionError):
                dsa_token_shard_restore_tokens(y, plan, plan.num_tokens_pad + 1)


class TestTokenShardPlumbing(CustomTestCase):
    """The non-collective helpers: top-k slicing and the cached query lens."""

    def test_slice_takes_this_ranks_rows(self):
        plan = plan_dsa_token_shard([17], [117], 4, 2)
        x = torch.arange(20 * 3).reshape(20, 3)
        out = dsa_token_shard_slice(x, plan)
        self.assertEqual(out.shape[0], 5)
        torch.testing.assert_close(out, x[10:15])

    def test_slice_zero_pads_a_short_tail(self):
        # Rank 3's slice is [15, 20) but only 17 tokens are real.
        plan = plan_dsa_token_shard([17], [117], 4, 3)
        x = torch.arange(20 * 3).reshape(20, 3)
        out = dsa_token_shard_slice(x, plan)
        self.assertEqual(out.shape[0], 5)
        torch.testing.assert_close(out[:2], x[15:17])
        torch.testing.assert_close(out[2:], x.new_zeros((3, 3)))

    def test_cumulative_query_lens_cached_on_the_batch(self):
        plan = plan_dsa_token_shard([5, 9], [105, 214], 2, 0)
        batch = types.SimpleNamespace()
        out = dsa_token_shard_cumulative_query_lens(batch, plan, torch.device("cpu"))
        self.assertEqual(out.tolist(), cumulative(plan.query_lens))
        self.assertIs(batch.npu_dsa_token_shard_cu_query_lens, out)
        again = dsa_token_shard_cumulative_query_lens(batch, plan, torch.device("cpu"))
        self.assertIs(again, out)


if __name__ == "__main__":
    unittest.main()
