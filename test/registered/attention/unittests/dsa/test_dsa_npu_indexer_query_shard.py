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

When the DSA token shard runs, ``_IndexerQueryShard.resolve`` skips the top-k
all-gather and keeps this rank's rows, but only if the token shard planned the
same rows.

Usage:
    python -m pytest test_dsa_npu_indexer_query_shard.py -v
    python test_dsa_npu_indexer_query_shard.py
"""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa import dsa_npu_indexer
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
    _build_verify_indexer_shard,
    _get_indexer_query_shard,
    _IndexerQueryShard,
    plan_indexer_query_shard,
)
from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import plan_dsa_token_shard
from sglang.srt.model_executor.forward_mode import ForwardMode
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


class TestLocalTopkWithTokenShard(CustomTestCase):
    """Keeping the local top-k is right only when attention reads exactly these
    rows. Without a token shard it reads full width; with a plan for other rows
    every query gets another token's top-k, and no shape error catches it."""

    def _resolve(self, prefix_lens, extend_lens, tp_size, tp_rank, plan_rank):
        total = sum(extend_lens)
        shard = _shard_for(
            prefix_lens, extend_lens, tp_size, tp_rank, _ceil_align(total, tp_size)
        )
        plan = None
        if plan_rank is not None:
            seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
            plan = plan_dsa_token_shard(extend_lens, seq_lens, tp_size, plan_rank)
        batch = types.SimpleNamespace(npu_dsa_token_shard_plan=plan)
        topk = torch.arange(shard.rows * 4).reshape(shard.rows, 4)
        gathered = []

        def fake_gather(_shard, t, num_tokens):
            # The real one is a collective.
            gathered.append(True)
            return t

        with mock.patch.object(_IndexerQueryShard, "gather", fake_gather):
            out = shard.resolve(topk, total, batch)
        return out, topk, batch.npu_indexer_topk_is_local, bool(gathered)

    def test_gathers_without_a_token_shard(self):
        _, _, local, gathered = self._resolve([0], [16384], 16, 3, plan_rank=None)
        self.assertFalse(local)
        self.assertTrue(gathered)

    def test_keeps_its_rows_when_the_token_shard_planned_them(self):
        for prefix_lens, extend_lens, tp_size in (
            ([0], [16385], 16),
            ([0, 2048, 4096], [5003, 4001, 6002], 16),
            ([500], [17], 3),
        ):
            for tp_rank in range(tp_size):
                with self.subTest(ext=extend_lens, tp=tp_size, rank=tp_rank):
                    out, topk, local, gathered = self._resolve(
                        prefix_lens, extend_lens, tp_size, tp_rank, tp_rank
                    )
                    self.assertTrue(local)
                    self.assertFalse(gathered)
                    self.assertIs(out, topk)

    def test_gathers_when_the_token_shard_planned_other_rows(self):
        _, _, local, gathered = self._resolve([0], [16384], 16, 3, plan_rank=5)
        self.assertFalse(local)
        self.assertTrue(gathered)


class TestFp8PcpIndexerQueryShard(CustomTestCase):
    def test_local_rows_are_quantized_with_matching_metadata(self):
        batch = types.SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=object(),
            dsa_cp_metadata_prepared=True,
            out_cache_loc=torch.arange(4),
        )
        metadata = types.SimpleNamespace(
            seq_lens_cpu_int=None,
            seq_lens=torch.tensor([101, 103, 501, 503], dtype=torch.int32),
            actual_seq_lengths_q=torch.arange(1, 5, dtype=torch.int32),
            block_tables=torch.tensor([[1, 2], [1, 2], [5, 6], [5, 6]]),
            indexer_valid_requests=None,
        )
        metadata.actual_seq_lengths_kv = metadata.seq_lens
        indexer = dsa_npu_indexer.DSANPUIndexerMixin()
        indexer.dsa_enable_prefill_cp = True
        indexer.hidden_size = indexer.n_heads = 1
        indexer.num_init_tokens = indexer.num_local_tokens = 0
        indexer.head_dim = 128
        indexer.index_topk = 2
        indexer.rotary_emb = None
        indexer.weights_proj = mock.Mock(
            side_effect=lambda x: (x,), weight=torch.empty(1, dtype=torch.bfloat16)
        )
        indexer.wq_b = indexer.wk = mock.Mock(
            side_effect=lambda x: (x.expand(-1, 128).contiguous(),)
        )
        indexer.k_norm = mock.Mock(side_effect=lambda x: x)
        cache = torch.empty((8, 16, 1, 128), dtype=torch.float8_e4m3fn)
        scales = torch.ones(8, 16, 1)
        pool = types.SimpleNamespace(
            index_k_scale_buffer=scales,
            indexer_hadamard_128=None,
            index_k_store_dtype=torch.float8_e4m3fn,
            index_k_scale_dtype=torch.float32,
            set_index_k_buffer=mock.Mock(),
            set_index_k_scale_buffer=mock.Mock(),
            get_index_k_buffer=lambda _: cache,
            get_index_k_scale_buffer=lambda _: scales,
        )
        inputs = torch.arange(4, dtype=torch.bfloat16).view(-1, 1)
        quantize = mock.Mock(side_effect=lambda x, *_: (x, x[..., 0].float()))
        op = mock.Mock(
            side_effect=lambda **kw: torch.zeros(
                (kw["query"].shape[0], 1, 2), dtype=torch.int32
            )
        )
        with (
            mock.patch.object(
                dsa_npu_indexer,
                "get_attn_backend",
                return_value=types.SimpleNamespace(forward_metadata=metadata),
            ),
            mock.patch.object(
                dsa_npu_indexer, "get_token_to_kv_pool", return_value=pool
            ),
            mock.patch.object(
                dsa_npu_indexer,
                "cp_gather_full_sequence_states",
                side_effect=lambda k, *_: k,
            ),
            mock.patch.object(
                dsa_npu_indexer, "_quantize_npu_indexer_activation", quantize
            ),
            mock.patch.object(
                dsa_npu_indexer.envs.SGLANG_NPU_USE_MULTI_STREAM,
                "get",
                return_value=False,
            ),
            mock.patch.object(
                torch,
                "npu",
                types.SimpleNamespace(current_stream=lambda: None),
                create=True,
            ),
            mock.patch.object(
                torch.ops.npu, "npu_quant_lightning_indexer", op, create=True
            ),
            get_parallel().override(attn_tp_size=2, attn_tp_rank=1, dcp_enabled=False),
        ):
            topk = indexer.forward_npu(
                inputs, inputs, torch.arange(4), batch, 0, allow_pcp_query_shard=True
            )
            self.assertEqual(tuple(topk.shape), (2, 2))
            shard = batch.npu_dsa_pcp_query_shard
            kwargs = op.call_args.kwargs
            torch.testing.assert_close(kwargs["query"][..., 0], inputs[2:])
            torch.testing.assert_close(kwargs["weights"], inputs[2:])
            torch.testing.assert_close(
                kwargs["query_dequant_scale"], inputs[2:].float()
            )
            self.assertIs(kwargs["key"], cache)
            self.assertIs(kwargs["key_dequant_scale"], scales)
            self.assertIs(kwargs["block_table"], shard.block_table)
            torch.testing.assert_close(shard.block_table, metadata.block_tables[2:])
            torch.testing.assert_close(
                shard.query_ends, metadata.actual_seq_lengths_q[:2]
            )
            torch.testing.assert_close(
                shard.key_lens, metadata.actual_seq_lengths_kv[2:]
            )
            self.assertEqual(quantize.call_args.args[0].shape[0], 2)

            topk = indexer.forward_npu(inputs, inputs, torch.arange(4), batch, 0)
            self.assertEqual(tuple(topk.shape), (4, 2))
            self.assertIsNone(batch.npu_dsa_pcp_query_shard)
            self.assertIs(op.call_args.kwargs["block_table"], metadata.block_tables)


class TestVerifyIndexerRequestShard(CustomTestCase):
    def test_whole_requests_and_rank_order(self):
        width, batch_size, tp_size = 5, 32, 16
        query = torch.arange(batch_size * width)
        query_ends = torch.arange(width, width * (batch_size + 1), width)
        kv_lens = torch.arange(batch_size) + 4096
        shards = [
            _build_verify_indexer_shard(
                query.numel(), width, query_ends, kv_lens, tp_size, rank
            )
            for rank in range(tp_size)
        ]
        for rank, shard in enumerate(shards):
            self.assertEqual(shard.rows, 10)
            self.assertEqual(shard.actual_seq_lengths_q.tolist(), [5, 10])
            self.assertEqual(
                shard.actual_seq_lengths_kv.tolist(), [4096 + 2 * rank, 4097 + 2 * rank]
            )
        self.assertTrue(torch.equal(torch.cat([s.take(query) for s in shards]), query))

    def test_incompatible_captured_shapes(self):
        for num_tokens, batch_size, kv_size in (
            (75, 15, 15),
            (79, 16, 16),
            (80, 16, 15),
        ):
            with self.subTest(
                num_tokens=num_tokens, batch_size=batch_size, kv_size=kv_size
            ):
                query_ends = torch.arange(1, batch_size + 1) * 5
                self.assertIsNone(
                    _build_verify_indexer_shard(
                        num_tokens, 5, query_ends, torch.ones(kv_size), 16, 0
                    )
                )


if __name__ == "__main__":
    unittest.main()
