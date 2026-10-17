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

Usage:
    python -m pytest test_dsa_token_shard_plan.py -v
    python test_dsa_token_shard_plan.py
"""

import itertools
import types
import unittest

from sglang.srt.layers.attention.dsa.dsa_token_shard import _build_dsa_token_shard_plan
from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import plan_dsa_token_shard
from sglang.srt.model_executor.forward_batch_info import ForwardMode
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


class TestTokenShardRefusesDcp(CustomTestCase):
    """DCP's DSA path all-gathers the query across the DCP group and expects
    every token's head-sharded row; a token slice would be merged wrong."""

    def _plan(self, dcp_enabled):
        batch = types.SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[16384],
            extend_prefix_lens_cpu=[0],
        )
        with get_parallel().override(
            attn_tp_size=16, attn_tp_rank=3, dcp_enabled=dcp_enabled
        ):
            return _build_dsa_token_shard_plan(batch)

    def test_planned_without_dcp(self):
        self.assertIsNotNone(self._plan(dcp_enabled=False))

    def test_refused_under_dcp(self):
        self.assertIsNone(self._plan(dcp_enabled=True))


if __name__ == "__main__":
    unittest.main()
