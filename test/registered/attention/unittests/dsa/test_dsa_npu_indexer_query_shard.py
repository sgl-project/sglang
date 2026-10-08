"""CPU unit test for the DSA indexer query shard, and for the gate that admits it.

``plan_indexer_query_shard`` (``layers/attention/dsa/dsa_npu_indexer.py``) splits a
prefill batch's indexer queries across the attention-TP group so each rank scores
only its own 1/tp of the rows. ``_get_indexer_query_shard`` decides whether that
plan describes the call in front of it.

The gate compares the query tensor's width with the plan. ``shard.total`` counts
real tokens; the width is padded up to a multiple of attn_tp_size for the MLP
reduce-scatter (``ForwardBatch.prepare_mlp_sync_batch``). An equality check
would turn sharding off for every token count that is not already a multiple of
attn_tp_size, so the gate admits any width in ``[total, rows * tp_size]``.

Usage:
    python -m pytest test_dsa_npu_indexer_query_shard.py -v
    python test_dsa_npu_indexer_query_shard.py
"""

import types
import unittest

import torch

from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
    _get_indexer_query_shard,
    plan_indexer_query_shard,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# Prefix-cached tails compute whatever the prompt leaves over; only 16384 (the
# chunked-prefill size) is a multiple of 16.
BOX_EXTENDS = [16384, 13846, 10828, 17941, 1557]


def _ceil_align(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


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
            start, rows, num_real, cum_query_lens, key_lens = plan_indexer_query_shard(
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

    def test_the_ranks_slices_reassemble_the_batch(self):
        # What the all-gather of per-rank top-k relies on: rank r's take() is
        # rows [r * rows, (r + 1) * rows) of the padded tensor.
        for total, tp_size in ((16384, 16), (10828, 16), (17, 16), (1557, 8)):
            padded = _ceil_align(total, tp_size)
            x = torch.arange(padded)
            parts = [
                _shard_for([500], [total], tp_size, r, padded).take(x)
                for r in range(tp_size)
            ]
            self.assertTrue(
                torch.equal(torch.cat(parts)[:total], x[:total]),
                f"total={total} tp_size={tp_size}",
            )


class TestIndexerQueryShardGate(CustomTestCase):
    """The gate must admit the padded query tensor the model actually passes."""

    def test_a_padded_width_is_admitted(self):
        # 10828 real tokens at attn_tp_size 16 arrive as a
        # ceil_align(10828, 16) = 10832-row query tensor.
        for total in BOX_EXTENDS:
            padded = _ceil_align(total, 16)
            shard = _shard_for([989184], [total], 16, 0, padded)
            self.assertIsNotNone(shard, f"total={total} padded={padded}")
            self.assertEqual(shard.total, total)
            self.assertEqual(shard.rows * shard.tp_size, padded)

    def test_a_width_outside_the_plan_is_refused(self):
        # Fewer rows than real tokens means the plan does not describe this
        # call; scoring every row is wrong but slicing it would be worse.
        self.assertIsNone(_shard_for([989184], [10828], 16, 0, 10827))
        self.assertIsNone(_shard_for([989184], [10828], 16, 0, 1024))
        # More rows than the plan covers: the gathered top-k would come back
        # shorter than the query tensor.
        self.assertIsNone(_shard_for([989184], [10828], 16, 0, 10833))


if __name__ == "__main__":
    unittest.main()
