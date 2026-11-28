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
from itertools import product
from unittest import mock

import torch

import sglang.srt.layers.attention.dsa.dsa_npu_indexer as npu_indexer
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
    _build_decode_indexer_shard,
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


class TestDecodeIndexerRequestShard(CustomTestCase):
    def test_whole_requests_and_rank_order(self):
        width, batch_size, tp_size = 5, 32, 16
        query = torch.arange(batch_size * width)
        query_ends = torch.arange(width, width * (batch_size + 1), width)
        kv_lens = torch.arange(batch_size) + 4096
        shards = [
            _build_decode_indexer_shard(width, query_ends, kv_lens, tp_size, rank)
            for rank in range(tp_size)
        ]
        for rank, shard in enumerate(shards):
            self.assertEqual(shard.rows, 10)
            self.assertEqual(shard.actual_seq_lengths_q.tolist(), [5, 10])
            self.assertEqual(
                shard.actual_seq_lengths_kv.tolist(), [4096 + 2 * rank, 4097 + 2 * rank]
            )
        self.assertTrue(torch.equal(torch.cat([s.take(query) for s in shards]), query))

    def test_uneven_shards_gather_fixed_tiles_in_request_order(self):
        width, batch_size, tp_size = 3, 5, 4
        topk = torch.arange(batch_size * width, dtype=torch.int32).view(-1, 1)
        query_ends = torch.arange(1, batch_size + 1) * width
        kv_lens = torch.arange(batch_size) + 4096
        padding = torch.full((3 * width, 1), 123456, dtype=topk.dtype)
        tiles = torch.cat((topk, padding)).chunk(tp_size)

        # TP4 DP alignment can pad the 15 query rows to 16.
        for dp_padding, rank in product((0, 1), range(tp_size)):
            with self.subTest(dp_padding=dp_padding, rank=rank):
                num_tokens = topk.shape[0] + dp_padding
                expected_tile = tiles[rank]
                shard = _build_decode_indexer_shard(
                    width, query_ends, kv_lens, tp_size, rank
                )
                local_topk = shard.take(topk)
                self.assertEqual(local_topk.shape[0], (6, 6, 3, 0)[rank])

                def gather(out, local):
                    self.assertEqual(local.shape, expected_tile.shape)
                    real_rows = shard.num_real
                    self.assertTrue(
                        torch.equal(local[:real_rows], expected_tile[:real_rows])
                    )
                    out.copy_(torch.cat(tiles))

                with mock.patch.object(
                    npu_indexer, "attn_tp_all_gather_into_tensor", side_effect=gather
                ) as collective:
                    output = shard.gather(local_topk, num_tokens)
                collective.assert_called_once()
                self.assertEqual(output.shape[0], num_tokens)
                self.assertTrue(torch.equal(output[: topk.shape[0]], topk))

    def _run_forward(
        self,
        mode,
        graph,
        dtype,
        request_slots,
        *,
        enabled=True,
        dp_query_padding=0,
        input_on_attn_tp_slices=False,
        ragged_verify_layout=None,
        expect_kernel=True,
    ):
        verify = mode.is_target_verify()
        width = (5 if graph else 3) if verify else 1
        bs = len(request_slots)
        tokens = bs * width + dp_query_padding
        activations = torch.arange(tokens, dtype=torch.bfloat16).view(-1, 1)
        lengths = torch.tensor(
            [128 + i if slot else 0 for i, slot in enumerate(request_slots)],
            dtype=torch.int32,
        )
        table = torch.arange(1, bs + 1, dtype=torch.int32).view(bs, 1)
        metadata = types.SimpleNamespace(
            seq_lens=lengths,
            seq_lens_cpu_int=None if graph else lengths,
            actual_seq_lengths_q=torch.arange(1, bs + 1) * width,
            block_tables=table,
        )
        backend = types.SimpleNamespace(
            forward_metadata=metadata,
            use_decode_indexer_request_shard=enabled,
        )
        batch = types.SimpleNamespace(
            forward_mode=mode,
            spec_info=types.SimpleNamespace(draft_token_num=width) if verify else None,
            ragged_verify_layout=ragged_verify_layout,
            attn_cp_metadata=None,
            req_pool_indices=torch.tensor(request_slots),
            out_cache_loc=torch.arange(tokens),
        )
        indexer = npu_indexer.DSANPUIndexerMixin()
        indexer.dsa_enable_prefill_cp = False
        indexer.hidden_size = indexer.n_heads = indexer.head_dim = (
            indexer.index_topk
        ) = 1
        indexer.num_init_tokens = indexer.num_local_tokens = 0
        indexer.rotary_emb = None
        indexer.weights_proj = mock.Mock(side_effect=lambda x: (x,), weight=activations)
        indexer.wq_b = indexer.wk = lambda x: (x,)
        indexer.k_norm = lambda x: x
        quantized = dtype != torch.bfloat16
        cache = torch.zeros((bs + 1, 16, 1, 1), dtype=dtype)
        scales = torch.zeros((bs + 1, 16, 1, 1), dtype=torch.float32)
        pool = types.SimpleNamespace(
            index_k_scale_buffer=scales if quantized else None,
            indexer_hadamard_128=None,
            index_k_store_dtype=dtype,
            index_k_scale_dtype=scales.dtype,
            get_index_k_buffer=lambda _: cache,
            get_index_k_scale_buffer=lambda _: scales,
            set_index_k_buffer=mock.Mock(),
            set_index_k_scale_buffer=mock.Mock(),
        )

        def topk(**kwargs):
            return kwargs["query"].to(torch.int32) + 1

        quant_kernel = mock.Mock(side_effect=topk)
        bf16_kernel = mock.Mock(side_effect=lambda **kwargs: (topk(**kwargs),))
        with (
            get_parallel().override(attn_tp_size=2, attn_tp_rank=1, dcp_enabled=False),
            mock.patch.object(npu_indexer, "get_attn_backend", return_value=backend),
            mock.patch.object(npu_indexer, "get_token_to_kv_pool", return_value=pool),
            mock.patch.object(
                envs.SGLANG_NPU_USE_MULTI_STREAM, "get", return_value=False
            ),
            mock.patch.object(npu_indexer, "_use_ag_after_qlora", False),
            mock.patch.object(
                npu_indexer,
                "torch_npu",
                types.SimpleNamespace(npu_lightning_indexer=bf16_kernel),
                create=True,
            ),
            mock.patch.object(
                torch.ops.npu, "npu_quant_lightning_indexer", quant_kernel, create=True
            ),
            mock.patch.object(
                npu_indexer,
                "_quantize_npu_indexer_activation",
                side_effect=lambda x, *_: (x, torch.ones_like(x, dtype=scales.dtype)),
            ),
            mock.patch.object(
                _IndexerQueryShard,
                "gather",
                autospec=True,
                side_effect=lambda _, local, __: local,
            ) as gather,
        ):
            output = indexer.forward_npu(
                activations,
                activations,
                torch.arange(tokens),
                batch,
                0,
                input_on_attn_tp_slices=input_on_attn_tp_slices,
            )
        kernel = quant_kernel if quantized else bf16_kernel
        self.assertFalse(batch.npu_indexer_topk_is_local)
        args = None
        if expect_kernel:
            kernel.assert_called_once()
            args = kernel.call_args.kwargs
            self.assertIs(args["key"], cache)
            if quantized:
                self.assertIs(args["key_dequant_scale"], scales)
                self.assertEqual(args["query_dequant_scale"].shape, args["query"].shape)
        else:
            quant_kernel.assert_not_called()
            bf16_kernel.assert_not_called()
        return output, args, gather, width

    def test_uneven_shards_only_score_local_requests(self):
        cases = product(
            (ForwardMode.TARGET_VERIFY, ForwardMode.DECODE),
            (False, True),
            (torch.bfloat16, torch.float8_e4m3fn, torch.int8),
            ((11,), (11, 12, 13)),
        )
        for mode, graph, dtype, slots in cases:
            with self.subTest(mode=mode, graph=graph, dtype=dtype, slots=slots):
                empty_rank = len(slots) == 1
                local, args, gather, width = self._run_forward(
                    mode, graph, dtype, slots, expect_kernel=not empty_rank
                )
                gather.assert_called_once()
                self.assertEqual(
                    gather.call_args.args[0].rows, (1 if empty_rank else 2) * width
                )
                self.assertEqual(gather.call_args.args[2], len(slots) * width)
                self.assertEqual(local.dtype, torch.int32)
                if empty_rank:
                    self.assertEqual(local.shape, (0, 1))
                else:
                    self.assertEqual(args["actual_seq_lengths_query"].tolist(), [width])
                    self.assertEqual(args["actual_seq_lengths_key"].tolist(), [130])
                    self.assertEqual(args["block_table"].flatten().tolist(), [3])
                    self.assertEqual(
                        local.flatten().tolist(),
                        list(range(2 * width + 1, 3 * width + 1)),
                    )

    def test_request_shard_preserves_indexer_inputs_and_outputs(self):
        cases = product(
            (ForwardMode.TARGET_VERIFY, ForwardMode.DECODE),
            (True, False),
            (torch.bfloat16, torch.float8_e4m3fn, torch.int8),
            ((11, 12, 13, 0), (11, 12, 0, 0)),
        )
        for mode, graph, dtype, slots in cases:
            with self.subTest(mode=mode, graph=graph, dtype=dtype, slots=slots):
                local, args, gather, width = self._run_forward(
                    mode, graph, dtype, slots
                )
                gather.assert_called_once()
                self.assertEqual(gather.call_args.args[2], 4 * width)
                self.assertEqual(
                    args["actual_seq_lengths_query"].tolist(), [width, 2 * width]
                )
                self.assertEqual(
                    args["actual_seq_lengths_key"].tolist(),
                    [130 if slots[2] else 0, 0],
                )
                self.assertEqual(args["block_table"].flatten().tolist(), [3, 4])
                self.assertEqual(
                    local.flatten().tolist(), list(range(2 * width + 1, 4 * width + 1))
                )

    def test_preserves_fallback_for_ineligible_inputs(self):
        slots = (11, 12, 13, 14)
        for mode, graph, options in product(
            (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY),
            (False, True),
            ({"input_on_attn_tp_slices": True}, {"enabled": False}),
        ):
            with self.subTest(mode=mode, graph=graph, options=options):
                output, args, gather, width = self._run_forward(
                    mode, graph, torch.bfloat16, slots, **options
                )
                gather.assert_not_called()
                self.assertEqual(args["query"].shape[0], len(slots) * width)
                self.assertEqual(output.shape[0], len(slots) * width)

    def test_ragged_verify_falls_back_even_when_total_rows_fit(self):
        for graph in (True, False):
            with self.subTest(graph=graph):
                output, args, gather, width = self._run_forward(
                    ForwardMode.TARGET_VERIFY,
                    graph,
                    torch.bfloat16,
                    (11, 12, 13, 14),
                    ragged_verify_layout=object(),
                )
                gather.assert_not_called()
                self.assertEqual(args["query"].shape[0], 4 * width)
                self.assertEqual(output.shape[0], 4 * width)

    def test_eager_verify_with_dp_query_padding_scores_assigned_requests(self):
        # For TP2, DP alignment rounds three 3-row requests from 9 rows to 10.
        local, args, gather, _ = self._run_forward(
            ForwardMode.TARGET_VERIFY,
            False,
            torch.bfloat16,
            (11, 12, 13),
            dp_query_padding=1,
        )
        gather.assert_called_once()
        self.assertEqual(gather.call_args.args[2], 10)
        self.assertEqual(args["actual_seq_lengths_query"].tolist(), [3])
        self.assertEqual(local.flatten().tolist(), [7, 8, 9])


if __name__ == "__main__":
    unittest.main()
