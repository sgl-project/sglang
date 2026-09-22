"""CPU unit test for the DSA-CP token-shard plan.

Pins ``plan_dsa_cp_shard`` (``layers/attention/dsa/dsa_cp_layout.py``). DSA-CP
cuts an extend batch's TOKENS across the attention-TP group, so a rank computes
every head for its slice instead of its own heads for every token. The cut is by
position, not by request, and at a 13,855-token tail over 16 ranks a request
straddles almost every boundary -- so each rank's per-request query and key
lengths have to be recomputed against its slice.

That recomputation is clamp arithmetic, and getting it wrong is silent: the
sparse operator is handed a span that is merely different rather than invalid,
reads real KV from the wrong place, and returns fluent text that stops following
the prompt. So the test does not check the clamps. It rebuilds the answer the
slow way -- walk every token, assign it to a rank, read the counts off -- and
requires the two to agree, over the served shapes, every small batch
exhaustively, and random large ragged ones.

Usage:
    python -m pytest test_dsa_cp_shard_plan.py -v
    python test_dsa_cp_shard_plan.py
"""

import itertools
import random
import unittest

from sglang.srt.layers.attention.dsa.dsa_cp_layout import plan_dsa_cp_shard
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


class TestDsaCpShardPlan(CustomTestCase):
    def _check(self, extend_seq_lens, seq_lens, tp_size, label=""):
        num_tokens = sum(extend_seq_lens)
        rows = -(-num_tokens // tp_size)
        covered = [0] * num_tokens

        for rank in range(tp_size):
            plan = plan_dsa_cp_shard(extend_seq_lens, seq_lens, tp_size, rank)
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

    def test_the_served_shapes(self):
        # p10's cached tail, AISBench's, and a chunked-prefill chunk.
        self._check([13855], [972319], 16, "p10 tail")
        self._check([10828], [1000012], 16, "aisbench tail")
        self._check([16384], [933888], 16, "prefill chunk")

    def test_a_key_length_is_the_last_tokens_not_the_first(self):
        # The operator walks back from the last query row (right-down causal),
        # so handing it the first token's span would truncate every other row.
        plan = plan_dsa_cp_shard([8], [108], 4, 0)
        self.assertEqual(plan.query_lens, [2])
        self.assertEqual(plan.key_lens, [102])  # prefix 100 + 2 tokens, not + 1

    def test_requests_outside_the_slice_are_dropped_entirely(self):
        # A length without a query row is a span the operator would read for
        # nothing, so it must be zero rather than the request's true length.
        plan = plan_dsa_cp_shard([2, 2, 2, 2], [102, 202, 302, 402], 4, 2)
        self.assertEqual(plan.query_lens, [0, 0, 2, 0])
        self.assertEqual(plan.key_lens, [0, 0, 302, 0])

    def test_every_small_batch_exhaustively(self):
        # 1-3 requests, 0-6 tokens each, tp 1-5. Off-by-ones live here.
        for nreq in (1, 2, 3):
            for lens in itertools.product(range(7), repeat=nreq):
                if sum(lens) == 0:
                    continue
                seqs = [n + 10 * (i + 1) for i, n in enumerate(lens)]
                for tp in (1, 2, 3, 4, 5):
                    self._check(list(lens), seqs, tp, f"{lens} tp{tp}")

    def test_random_large_ragged_batches(self):
        rng = random.Random(20260918)
        for trial in range(200):
            nreq = rng.randint(1, 6)
            lens = [rng.randint(0, 40000) for _ in range(nreq)]
            if sum(lens) == 0:
                continue
            seqs = [n + rng.randint(0, 1_000_000) for n in lens]
            self._check(lens, seqs, rng.choice([2, 4, 8, 16]), f"rand {trial}")

    def test_fewer_tokens_than_ranks(self):
        # Legal to plan, even though the runtime gate declines it: ranks past
        # the tokens get an empty slice rather than a negative one.
        plan = plan_dsa_cp_shard([3], [103], 16, 9)
        self.assertEqual(plan.rows, 1)
        self.assertEqual(plan.num_local_tokens, 0)
        self.assertTrue(plan.is_empty())
        self.assertEqual(plan.query_lens, [0])

    def test_mismatched_metadata_raises_rather_than_guesses(self):
        with self.assertRaises(AssertionError):
            plan_dsa_cp_shard([5, 9], [105], 4, 0)
        with self.assertRaises(AssertionError):
            plan_dsa_cp_shard([5], [105], 4, 4)


if __name__ == "__main__":
    unittest.main()
