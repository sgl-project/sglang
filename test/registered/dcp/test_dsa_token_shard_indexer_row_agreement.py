"""CPU unit test: Stage A and the DSA token shard must shard an extend batch the same way.

Two planners, written separately, in modules that do not import each other:

* ``plan_indexer_query_shard`` (``layers/attention/dsa/dsa_npu_indexer.py``)
  picks which indexer query rows a rank scores.
* ``plan_dsa_token_shard`` (``layers/attention/dsa/dsa_token_shard_layout.py``) picks which
  attention query rows the same rank computes.

Both take ``ceil(sum(extend_lens) / tp_size)`` rows starting at
``tp_rank * rows``. That is a coincidence of two implementations, not a shared
function, and nothing in the tree says it has to hold.

**Today it does not have to.** Stage A all-gathers the top-k back to full width
before anything reads it, so the DSA token shard then slices that full-width tensor with its
own plan and Stage A's row choice cannot be observed.

**The handoff's W2 removes that all-gather** -- when the DSA token shard runs, each rank
already holds exactly the top-k rows it is about to attend, so the gather sends
``(tp-1)/tp`` of the data to be discarded. Dropping it makes the agreement
load-bearing: the rows one planner produced are consumed as the rows the other
planner designated, with no full-width tensor in between to hide a mismatch.

A mismatch would then be silent in the worst way. Every shape stays valid, the
operator runs, and each query attends a top-k list computed for a *different
token*. The output is fluent and wrong -- the same failure mode the the DSA token shard shard
plan's own test was written to catch.

So this pins the agreement before the optimisation depends on it, and it checks
the whole split rather than the row range alone: per-request query counts and
per-request key lengths too, since both features hand those to operators.

Usage:
    python -m pytest test_dsa_token_shard_indexer_row_agreement.py -v
    python test_dsa_token_shard_indexer_row_agreement.py
"""

import itertools
import random
import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa import dsa_token_shard as dsa_token_shard_module
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
    _IndexerQueryShard,
    plan_indexer_query_shard,
)
from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import plan_dsa_token_shard
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
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


def _num_local_tokens(plan):
    """Real tokens in the slice: ``rows`` minus its padding, 0 past the tokens."""
    return max(0, plan.local_end - plan.local_start)


class TestStageAAndDsaTokenShardAgree(CustomTestCase):
    def _check(self, prefix_lens, extend_lens, tp_size, label=""):
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]

        for tp_rank in range(tp_size):
            start, rows, num_real, cum_query_lens, key_lens = plan_indexer_query_shard(
                prefix_lens, extend_lens, tp_size, tp_rank
            )
            plan = plan_dsa_token_shard(extend_lens, seq_lens, tp_size, tp_rank)

            where = f"{label} rank={tp_rank}/{tp_size}"

            # The row range. This is what W2 makes load-bearing: the rows Stage A
            # scored are consumed as the rows the DSA token shard expected.
            self.assertEqual(start, plan.local_start, f"{where} start")
            self.assertEqual(rows, plan.rows, f"{where} rows")
            self.assertEqual(
                num_real, _num_local_tokens(plan), f"{where} real token count"
            )

            # The per-request split. Stage A keeps it cumulative and the DSA token shard keeps
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
        # the DSA token shard's runtime gate declines this, but the planners still have to
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
                    plan_dsa_token_shard([total], [total], tp_size, r)
                    for r in range(tp_size)
                ]
                padded = [
                    r for r, p in enumerate(plans) if _num_local_tokens(p) < p.rows
                ]
                # Ranks that hold no real token at all are empty, not padded;
                # both are "not full", and neither may precede a full rank.
                full = [
                    r for r, p in enumerate(plans) if _num_local_tokens(p) == p.rows
                ]
                if full and padded:
                    self.assertLess(
                        max(full),
                        min(padded),
                        f"total={total} tp={tp_size}: a padded rank precedes a full one",
                    )


class _FakeBatch:
    """Just enough ForwardBatch for ``get_dsa_token_shard_plan`` to read its cache."""

    def __init__(self, plan):
        self.npu_dsa_token_shard_plan = plan


def _shard_for(extend_lens, prefix_lens, tp_size, tp_rank):
    start, rows, num_real, _, _ = plan_indexer_query_shard(
        prefix_lens, extend_lens, tp_size, tp_rank
    )
    return _IndexerQueryShard(
        start=start,
        rows=rows,
        num_real=num_real,
        total=sum(extend_lens),
        tp_size=tp_size,
        actual_seq_lengths_q=None,
        actual_seq_lengths_kv=None,
    )


class TestW2LocalTopk(CustomTestCase):
    """``_IndexerQueryShard.resolve`` -- handoff §8 W2.

    It drops the top-k all-gather when the DSA token shard is about to slice the result back
    to the rows this rank already holds. The decision has to be exactly right:
    taking the local path when the DSA token shard is NOT running hands attention a tensor
    a sixteenth of the width it expects, and skipping it when the two planners
    disagree hands every query a top-k computed for a different token, with no
    shape error to catch it.
    """

    TOPK = 8  # stands in for index_topk; only the row axis matters here

    def _resolve(self, extend_lens, prefix_lens, tp_size, tp_rank, plan_rank=None):
        """Returns (out, flag, gathered, shard). ``plan_rank`` skews the plan."""
        shard = _shard_for(extend_lens, prefix_lens, tp_size, tp_rank)
        plan = (
            None
            if plan_rank is None
            else plan_dsa_token_shard(
                extend_lens,
                [p + e for p, e in zip(prefix_lens, extend_lens)],
                tp_size,
                plan_rank,
            )
        )
        batch = _FakeBatch(plan)
        topk = (
            torch.arange(shard.rows * self.TOPK, dtype=torch.int32).reshape(
                shard.rows, self.TOPK
            )
            + 1
        )
        seen = {"gathered": False}

        def _fake_gather(_self, t, num_tokens):
            # patch.object on the CLASS leaves this unbound, so it is handed
            # the shard as well -- the collective itself needs a process group.
            seen["gathered"] = True
            return t

        # The flag is a cached function now (merge blocker 8), so patch the
        # function rather than a module global.
        with (
            mock.patch.object(
                dsa_token_shard_module, "_dsa_token_shard_flag", lambda: True
            ),
            mock.patch.object(_IndexerQueryShard, "gather", _fake_gather),
        ):
            out = shard.resolve(topk, sum(extend_lens), batch)
        return out, batch.npu_indexer_topk_is_local, seen["gathered"], shard

    def test_gathers_when_dsa_token_shard_is_not_running(self):
        """No plan means attention reads full width, so the gather must happen."""
        out, local, gathered, _ = self._resolve([16384], [0], 16, 3, plan_rank=None)
        self.assertFalse(local)
        self.assertTrue(gathered, "dropped the gather with the DSA token shard off")

    def test_skips_the_gather_when_dsa_token_shard_planned_the_same_rows(self):
        for tp_size in TP_SIZES:
            for tp_rank in range(tp_size):
                for extend_lens, prefix_lens in (
                    ([16384], [0]),
                    ([13855], [4096]),
                    ([1557], [1_000_000]),
                    ([5003, 4001, 6002], [0, 2048, 4096]),
                ):
                    with self.subTest(tp=tp_size, rank=tp_rank, ext=extend_lens):
                        out, local, gathered, shard = self._resolve(
                            extend_lens,
                            prefix_lens,
                            tp_size,
                            tp_rank,
                            plan_rank=tp_rank,
                        )
                        self.assertTrue(local)
                        self.assertFalse(gathered)
                        self.assertEqual(out.shape[0], shard.rows)

    def test_padding_rows_are_zeroed_exactly_as_the_gather_path_leaves_them(self):
        """``dsa_token_shard_slice`` zero-pads, so the local path must too -- bitwise.

        Two shapes that actually produce padding, from the arithmetic rather
        than from assumption:

        * 16385 over tp 16 -> rows 1025; only rank 15 is padded, by 15 rows.
          One chunked-prefill batch plus a single token, the everyday ragged
          case.
        * 17 over tp 16 -> rows 2; rank 8 holds 1 real row and ranks 9-15 hold
          no real token at all. An all-padding rank must come back all zeros,
          not as whatever the operator left in a top-k it never wrote.

        A batch of exactly 16384 pads no rank at all, which is why that shape
        once hid an alignment bug. It is covered by the test above, not here.
        """
        for extend_lens, prefix_lens, tp_size in (
            ([16385], [0], 16),
            ([17], [0], 16),
        ):
            for tp_rank in range(tp_size):
                with self.subTest(total=sum(extend_lens), rank=tp_rank):
                    out, local, _, shard = self._resolve(
                        extend_lens, prefix_lens, tp_size, tp_rank, plan_rank=tp_rank
                    )
                    self.assertTrue(local)
                    self.assertTrue(
                        bool((out[: shard.num_real] != 0).all()),
                        "zeroed a real row",
                    )
                    self.assertEqual(
                        int(out[shard.num_real :].abs().sum()),
                        0,
                        f"rank {tp_rank} left "
                        f"{shard.rows - shard.num_real} padding rows unzeroed; "
                        "the gather path would have zeroed them",
                    )

    def test_falls_back_to_the_gather_when_the_planners_disagree(self):
        """Unreachable while the agreement above holds -- and it must stay safe.

        Simulated by handing the shard a plan built for a different rank, which
        is exactly the shape a row-range disagreement would take.
        """
        out, local, gathered, _ = self._resolve([16384], [0], 16, 3, plan_rank=5)
        self.assertFalse(local, "used a plan that designates other rows")
        self.assertTrue(gathered)


class TestFlagsAreReachable(CustomTestCase):
    """Merge blocker 8: the feature flags were read at import.

    ``_enable_dsa_token_shard = envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD.get()`` ran before any test
    could set the variable, so the off path was unreachable from a test and the
    on path was whatever the CI environment happened to have. Both are now cached
    functions with an explicit reset, and this is what proves it.
    """

    def test_dsa_token_shard_flag_follows_the_environment_after_a_reset(self):
        from sglang.srt.environ import envs

        # MLAPO pinned off: with both set the flag deliberately raises, and
        # that is a different test from this one.
        try:
            for value in (False, True):
                with (
                    envs.SGLANG_NPU_USE_MLAPO.override(False),
                    envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD.override(value),
                ):
                    dsa_token_shard_module.reset_dsa_token_shard_flags()
                    self.assertEqual(
                        dsa_token_shard_module._dsa_token_shard_flag(), value
                    )
        finally:
            dsa_token_shard_module.reset_dsa_token_shard_flags()

    def test_the_flag_is_still_resolved_only_once_between_resets(self):
        """Cached on purpose: it decides whether a module is built at load time."""
        from sglang.srt.environ import envs

        dsa_token_shard_module.reset_dsa_token_shard_flags()
        with envs.SGLANG_NPU_USE_MLAPO.override(False):
            with envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD.override(True):
                first = dsa_token_shard_module._dsa_token_shard_flag()
            with envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD.override(False):
                self.assertEqual(
                    dsa_token_shard_module._dsa_token_shard_flag(),
                    first,
                    "the flag changed mid-run without a reset",
                )
        dsa_token_shard_module.reset_dsa_token_shard_flags()


class TestPlanRefusesDcp(CustomTestCase):
    """DCP's DSA path all-gathers the query across the DCP group, expecting
    every token's head-sharded row; a token-sliced query would be merged wrong."""

    def _plan(self, dcp_enabled):
        batch = types.SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[16384],
            extend_prefix_lens_cpu=[0],
        )
        with get_parallel().override(
            attn_tp_size=16, attn_tp_rank=3, dcp_enabled=dcp_enabled
        ):
            return dsa_token_shard_module._build_dsa_token_shard_plan(batch, 2048)

    def test_planned_without_dcp(self):
        self.assertIsNotNone(self._plan(dcp_enabled=False))

    def test_refused_under_dcp(self):
        self.assertIsNone(self._plan(dcp_enabled=True))


if __name__ == "__main__":
    unittest.main()
