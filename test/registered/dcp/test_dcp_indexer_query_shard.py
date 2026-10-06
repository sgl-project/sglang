"""CPU unit test for the DSA indexer query shard, and for the gate that admits it.

``plan_indexer_query_shard`` (``layers/attention/dsa/dsa_npu_indexer.py``) splits a
prefill batch's indexer queries across the attention-TP group so each rank scores
only its own 1/tp of the rows. ``_get_indexer_query_shard`` decides whether that
plan describes the call in front of it.

**The regression this file exists for.** The gate used to be
``shard.total != num_tokens``. ``shard.total`` counts real tokens; ``num_tokens``
is the width of the query tensor, which SGLang pads up to a multiple of
attn_tp_size for the MLP reduce-scatter (``ForwardBatch.prepare_mlp_sync_batch``).
Those are equal only when the token count is *already* a multiple of attn_tp_size,
so sharding silently switched off for 15 token counts in 16 -- no error, no log,
just 16x the indexer. It survived because every test and every probe used 16384:
the chunked-prefill size, and a multiple of 16. Prefix-cached tails compute
whatever the prompt leaves over, measured at 10,828 and 13,846 on the A3 box,
and got nothing. Fixed by admitting any width in ``[total, rows * tp_size]``.

All of this is index arithmetic -- no operator, no NPU, no collective -- so the
partition properties are asserted over *all* ranks together. A plan that drops a
row, double-counts one, or lands it a shard over passes every per-rank spot check
and still loses tokens; only the union catches it.

Usage:
    python -m pytest test_dcp_indexer_query_shard.py -v
    python test_dcp_indexer_query_shard.py
"""

import types
import unittest

import torch

from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
    _get_indexer_query_shard,
    _IndexerQueryShard,
    plan_indexer_query_shard,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

TP_SIZES = [2, 3, 4, 8, 16]

# Token counts measured on the A3 box. 16384 is the chunked-prefill size and the
# only one of these that is a multiple of 16 -- which is exactly why the bug hid.
BOX_EXTENDS = [16384, 13846, 10828, 17941, 1557]


def _ceil_align(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _plan(prefix_lens, extend_lens, tp_size, tp_rank):
    return plan_indexer_query_shard(prefix_lens, extend_lens, tp_size, tp_rank)


def _fake_batch(prefix_lens, extend_lens):
    """The attributes ``_build_indexer_query_shard`` actually reads."""
    return types.SimpleNamespace(
        extend_prefix_lens_cpu=list(prefix_lens),
        extend_seq_lens_cpu=list(extend_lens),
        forward_mode=ForwardMode.EXTEND,
        seq_lens=torch.tensor(
            [p + e for p, e in zip(prefix_lens, extend_lens)], dtype=torch.int32
        ),
    )


def _shard_for(prefix_lens, extend_lens, tp_size, tp_rank, num_tokens):
    batch = _fake_batch(prefix_lens, extend_lens)
    with get_parallel().override(attn_tp_size=tp_size, attn_tp_rank=tp_rank):
        return _get_indexer_query_shard(batch, num_tokens)


class TestIndexerQueryShardPlan(CustomTestCase):
    def test_the_padded_width_equals_the_plans_own_width(self):
        # The load-bearing identity behind the fix: rows * tp_size is exactly
        # ceil_align(total, tp_size), which is the padding SGLang already
        # applies. So the plan covers the padded tensor with nothing left over,
        # and the gate can admit it without any new arithmetic.
        for tp_size in TP_SIZES:
            for total in BOX_EXTENDS + [1, 2, 15, 16, 17, 97, 4096]:
                _, rows, _, _, _ = _plan([0], [total], tp_size, 0)
                self.assertEqual(
                    rows * tp_size,
                    _ceil_align(total, tp_size),
                    f"tp_size={tp_size} total={total}",
                )

    def test_the_ranks_partition_every_real_row_exactly_once(self):
        for tp_size in TP_SIZES:
            for extend_lens in ([16384], [10828], [1557, 10828], [1, 1, 1]):
                total = sum(extend_lens)
                prefix_lens = [1000 * (i + 1) for i in range(len(extend_lens))]
                covered = []
                for tp_rank in range(tp_size):
                    start, rows, num_real, _, _ = _plan(
                        prefix_lens, extend_lens, tp_size, tp_rank
                    )
                    self.assertEqual(start, tp_rank * rows)
                    covered.extend(range(start, start + num_real))
                self.assertEqual(
                    covered,
                    list(range(total)),
                    f"tp_size={tp_size} extend_lens={extend_lens}",
                )

    def test_per_request_counts_and_key_lengths(self):
        # key_lens is what sparse_mode=3's right-down causal crop reads: a
        # request's keys end at its last LOCAL token, so prefix + local_end.
        # A request with no row on this rank gets 0, not its prefix.
        prefix_lens = [989184, 500]
        extend_lens = [10828, 12]
        tp_size = 16
        total = sum(extend_lens)
        seen = 0
        for tp_rank in range(tp_size):
            start, rows, num_real, cum_query_lens, key_lens = _plan(
                prefix_lens, extend_lens, tp_size, tp_rank
            )
            self.assertEqual(cum_query_lens[-1], num_real)
            seen += num_real
            for i, key_len in enumerate(key_lens):
                local = cum_query_lens[i] - (cum_query_lens[i - 1] if i else 0)
                if local == 0:
                    self.assertEqual(key_len, 0)
                else:
                    self.assertGreater(key_len, prefix_lens[i])
                    self.assertLessEqual(key_len, prefix_lens[i] + extend_lens[i])
        self.assertEqual(seen, total)


class TestIndexerQueryShardGate(CustomTestCase):
    """The gate must admit the padded query tensor the model actually passes."""

    def test_a_padded_width_is_admitted(self):
        # THE REGRESSION. 10828 real tokens at attn_tp_size 16 arrive as a
        # ceil_align(10828, 16) = 10832-row query tensor. The old gate compared
        # 10828 != 10832 and silently scored every row on every rank.
        for total in BOX_EXTENDS:
            padded = _ceil_align(total, 16)
            shard = _shard_for([989184], [total], 16, 0, padded)
            self.assertIsNotNone(shard, f"total={total} padded={padded}")
            self.assertEqual(shard.total, total)
            self.assertEqual(shard.rows * shard.tp_size, padded)

    def test_a_query_tensor_shorter_than_the_real_tokens_is_refused(self):
        # Fewer rows than real tokens means the plan does not describe this
        # call; scoring every row is wrong but slicing it would be worse.
        self.assertIsNone(_shard_for([989184], [10828], 16, 0, 10827))
        self.assertIsNone(_shard_for([989184], [10828], 16, 0, 1024))


def _row_wise_linear(x, weight):
    """``x @ weight.T`` with each row's sum order fixed by K alone.

    CPU BLAS picks kernels by row count, so ``x[a:b] @ w.T`` is not bitwise
    ``(x @ w.T)[a:b]`` for ragged counts (MKL: 17, 195, 1025 rows differ). These
    tests pin the slice algebra, not a BLAS property.
    """
    return (x.unsqueeze(-2) * weight).sum(-1)


def _bare_shard(extend_lens, prefix_lens, tp_size, tp_rank):
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


class TestEarlySlice(CustomTestCase):
    """Projecting the slice must equal slicing the projection.

    The indexer calls ``shard.take`` before ``wq_b`` and the rotary, not after,
    so it projects ``1/tp`` of the rows instead of all of them. That is only
    sound because every output row of a Linear depends on its own input row
    alone -- and, for the padding rows the last rank carries, because ``wq_b``
    has ``bias=False``. With a bias the padded rows would come back as the bias
    instead of as zero, and the change would stop being bitwise.
    """

    def test_take_commutes_with_a_bias_free_linear(self):
        torch.manual_seed(0)
        weight = torch.randn(16, 12, dtype=torch.float64)
        for total, tp_size in ((16384, 16), (16385, 16), (17, 16), (1557, 8)):
            x = torch.randn(total, 12, dtype=torch.float64)
            for tp_rank in range(tp_size):
                with self.subTest(total=total, tp=tp_size, rank=tp_rank):
                    shard = _bare_shard([total], [0], tp_size, tp_rank)
                    late = shard.take(_row_wise_linear(x, weight))
                    early = _row_wise_linear(shard.take(x), weight)
                    self.assertTrue(
                        torch.equal(late, early),
                        "projecting this rank's rows differs from slicing the "
                        "full projection",
                    )

    def test_a_bias_would_break_it_on_the_padded_rank(self):
        """Documents the constraint rather than trusting it stays true.

        If someone gives ``wq_b`` a bias, the early slice stops being bitwise
        and this is the shape of the damage: only the padded rank, only its
        padded rows.
        """
        torch.manual_seed(0)
        weight = torch.randn(16, 12, dtype=torch.float64)
        bias = torch.randn(16, dtype=torch.float64)
        total, tp_size = 16385, 16
        x = torch.randn(total, 12, dtype=torch.float64)
        broke = [
            r
            for r in range(tp_size)
            for shard in [_bare_shard([total], [0], tp_size, r)]
            if not torch.equal(
                shard.take(_row_wise_linear(x, weight) + bias),
                _row_wise_linear(shard.take(x), weight) + bias,
            )
        ]
        self.assertEqual(
            broke,
            [15],
            "expected a bias to corrupt exactly the one rank that holds padding",
        )


class TestFlagIsReachable(CustomTestCase):
    """The flag is a cached function with a reset, not read at import, so a
    test can reach both paths."""

    def test_indexer_shard_flag_follows_the_environment_after_a_reset(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsa import dsa_npu_indexer

        try:
            for value in (False, True):
                with envs.SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING.override(value):
                    dsa_npu_indexer.reset_indexer_shard_flag()
                    self.assertEqual(dsa_npu_indexer._shard_indexer_queries(), value)
        finally:
            dsa_npu_indexer.reset_indexer_shard_flag()


if __name__ == "__main__":
    unittest.main()
