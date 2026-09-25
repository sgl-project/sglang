"""CPU unit test: Stage A and DSA-CP must shard an extend batch the same way.

Two planners, written separately, in modules that do not import each other:

* ``plan_indexer_query_shard`` (``layers/attention/dsa/dsa_npu_indexer.py``)
  picks which indexer query rows a rank scores.
* ``plan_dsa_cp_shard`` (``layers/attention/dsa/dsa_cp_layout.py``) picks which
  attention query rows the same rank computes.

Both take ``ceil(sum(extend_lens) / tp_size)`` rows starting at
``tp_rank * rows``. That is a coincidence of two implementations, not a shared
function, and nothing in the tree says it has to hold.

**Today it does not have to.** Stage A all-gathers the top-k back to full width
before anything reads it, so DSA-CP then slices that full-width tensor with its
own plan and Stage A's row choice cannot be observed.

**The handoff's W2 removes that all-gather** -- when DSA-CP runs, each rank
already holds exactly the top-k rows it is about to attend, so the gather sends
``(tp-1)/tp`` of the data to be discarded. Dropping it makes the agreement
load-bearing: the rows one planner produced are consumed as the rows the other
planner designated, with no full-width tensor in between to hide a mismatch.

A mismatch would then be silent in the worst way. Every shape stays valid, the
operator runs, and each query attends a top-k list computed for a *different
token*. The output is fluent and wrong -- the same failure mode the DSA-CP shard
plan's own test was written to catch.

So this pins the agreement before the optimisation depends on it, and it checks
the whole split rather than the row range alone: per-request query counts and
per-request key lengths too, since both features hand those to operators.

Usage:
    python -m pytest test_dsa_cp_indexer_row_agreement.py -v
    python test_dsa_cp_indexer_row_agreement.py
"""

import itertools
import random
import unittest

from sglang.srt.layers.attention.dsa.dsa_cp_layout import plan_dsa_cp_shard
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import plan_indexer_query_shard
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TP_SIZES = [2, 3, 4, 8, 16]

# Shapes measured on the A3 and A5 boxes: the chunked-prefill size, a p10 tail,
# the AISBench tail, a ragged computed tail, and a short prompt. Only 16384 is a
# multiple of 16, which is why an alignment bug hid behind it once before.
BOX_EXTENDS = [16384, 13855, 10828, 17941, 1557]


def _cumulative(values):
    out, total = [], 0
    for v in values:
        total += v
        out.append(total)
    return out


class TestStageAAndDsaCpAgree(CustomTestCase):
    def _check(self, prefix_lens, extend_lens, tp_size, label=""):
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]

        for tp_rank in range(tp_size):
            start, rows, num_real, cum_query_lens, key_lens = plan_indexer_query_shard(
                prefix_lens, extend_lens, tp_size, tp_rank
            )
            plan = plan_dsa_cp_shard(extend_lens, seq_lens, tp_size, tp_rank)

            where = f"{label} rank={tp_rank}/{tp_size}"

            # The row range. This is what W2 makes load-bearing: the rows Stage A
            # scored are consumed as the rows DSA-CP expected.
            self.assertEqual(start, plan.local_start, f"{where} start")
            self.assertEqual(rows, plan.rows, f"{where} rows")
            self.assertEqual(
                num_real, plan.num_local_tokens, f"{where} real token count"
            )

            # The per-request split. Stage A keeps it cumulative and DSA-CP keeps
            # it per-request, so compare in one form.
            self.assertEqual(
                cum_query_lens,
                _cumulative(plan.query_lens),
                f"{where} per-request query counts",
            )

            # The per-request key lengths. Both hand these to an operator, and
            # both must describe the same last-token-in-this-slice.
            self.assertEqual(
                key_lens, plan.key_lens, f"{where} per-request key lengths"
            )

    def test_the_shapes_the_boxes_actually_run(self):
        for extend in BOX_EXTENDS:
            for tp_size in TP_SIZES:
                self._check([989184], [extend], tp_size, f"cached tail {extend}")
                self._check([0], [extend], tp_size, f"cold prompt {extend}")

    def test_the_multi_request_batch_the_lift_admits(self):
        # p13's shape: three long-prefix tails in one forward.
        self._check([989184] * 3, [4096, 4096, 4096], 16, "p13 batch")
        # Ragged, so requests straddle slice boundaries on most ranks.
        self._check([12000, 3000, 70000], [1557, 13855, 4096], 8, "ragged batch")

    def test_every_small_batch_exhaustively(self):
        # 1-3 requests, 0-6 tokens each, tp 1-5. Off-by-ones live here, and a
        # ragged total means the last rank holds padding.
        for nreq in (1, 2, 3):
            for extend_lens in itertools.product(range(7), repeat=nreq):
                if sum(extend_lens) == 0:
                    continue
                prefix_lens = [10 * (i + 1) for i in range(nreq)]
                for tp_size in (1, 2, 3, 4, 5):
                    self._check(
                        prefix_lens,
                        list(extend_lens),
                        tp_size,
                        f"{extend_lens} tp{tp_size}",
                    )

    def test_random_large_ragged_batches(self):
        rng = random.Random(20260925)
        for trial in range(200):
            nreq = rng.randint(1, 6)
            extend_lens = [rng.randint(0, 40000) for _ in range(nreq)]
            if sum(extend_lens) == 0:
                continue
            prefix_lens = [rng.randint(0, 1_000_000) for _ in extend_lens]
            self._check(
                prefix_lens,
                extend_lens,
                rng.choice(TP_SIZES),
                f"rand {trial}",
            )

    def test_fewer_tokens_than_ranks(self):
        # DSA-CP's runtime gate declines this, but the planners still have to
        # agree: a rank past the tokens gets an empty slice, not a negative one.
        self._check([100], [3], 16, "3 tokens over 16 ranks")

    def test_only_the_last_rank_can_hold_padding(self):
        """Padding never lands in the middle of the group.

        W2's fix has to zero the rows past ``num_real`` when it returns a rank's
        own top-k instead of gathering, because the indexer computed those rows
        from a zeroed query and its output there is not zero. This bounds that
        concern: at most one rank is ever affected, and it is always the last.
        """
        for total in (1, 15, 16, 17, 1557, 13855, 16384):
            for tp_size in TP_SIZES:
                plans = [
                    plan_dsa_cp_shard([total], [total], tp_size, r)
                    for r in range(tp_size)
                ]
                padded = [r for r, p in enumerate(plans) if p.num_local_tokens < p.rows]
                # Ranks that hold no real token at all are empty, not padded;
                # both are "not full", and neither may precede a full rank.
                full = [r for r, p in enumerate(plans) if p.num_local_tokens == p.rows]
                if full and padded:
                    self.assertLess(
                        max(full),
                        min(padded),
                        f"total={total} tp={tp_size}: a padded rank precedes a full one",
                    )


if __name__ == "__main__":
    unittest.main()
