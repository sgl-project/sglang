import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.srt.layers.attention.dsa import kpool_fp8_index
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    _topk_from_pooled_history_logits_unfused,
    topk_from_pooled_history_logits,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKPoolMqaBackend(CustomTestCase):
    def _run_paged_mqa_dispatch(self, forward_mode, metadata_rows=None):
        batch_size = 2
        next_n = 6
        num_heads = 32
        n_real = batch_size * next_n
        metadata_rows = n_real if metadata_rows is None else metadata_rows
        num_q_padded = n_real + 4
        max_seq_len = 128

        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.index_kpool = 4
        indexer.index_topk = 8
        indexer.sm_count = 78
        indexer._get_index_k_read_buffer = MagicMock(
            return_value=torch.empty(2, 64 * 132, dtype=torch.uint8)
        )
        indexer._get_kpool_decode_metadata = MagicMock()
        indexer._should_use_tilelang_paged_mqa_logits = MagicMock(return_value=False)
        indexer._kpool_fused_topk_mapping = MagicMock(
            return_value=(None, None, None)
        )
        topk_result = object()
        indexer._topk_from_kpool_logits = MagicMock(return_value=topk_result)

        pool_seqlens = torch.arange(1, n_real + 1, dtype=torch.int32)
        pool_context_lens = pool_seqlens.view(-1, 1)
        pool_block_tables = torch.arange(n_real * 2, dtype=torch.int32).view(
            n_real, 2
        )
        stale_schedule = object()
        indexer._get_kpool_decode_metadata.return_value = (
            pool_seqlens,
            pool_context_lens,
            pool_block_tables,
            stale_schedule,
        )

        plan = SimpleNamespace(num_draft_tokens=next_n)
        metadata_block_tables = torch.arange(
            metadata_rows * 2, dtype=torch.int32
        ).view(metadata_rows, 2)
        metadata = SimpleNamespace(
            attn_metadata=SimpleNamespace(kpool_write_plan=plan),
            get_page_table_64=lambda: metadata_block_tables,
            get_seqlens_expanded=lambda: torch.arange(
                metadata_rows, dtype=torch.int32
            ),
            get_seqlens_int32=lambda: torch.arange(
                batch_size, dtype=torch.int32
            ),
        )
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode, num_token_non_padded_cpu=n_real
        )
        q_fp8 = torch.empty(
            num_q_padded, num_heads, 128, dtype=torch.float8_e4m3fn
        )
        weights = torch.empty(num_q_padded, num_heads, 1)
        pool = SimpleNamespace(page_size=64)
        schedule = object()
        deep_gemm = MagicMock()
        deep_gemm.get_paged_mqa_logits_metadata.return_value = schedule
        deep_gemm.fp8_paged_mqa_logits.return_value = torch.empty(
            n_real, max_seq_len
        )

        tilelang_logits = torch.empty(n_real, max_seq_len)
        with (
            patch.object(dsa_indexer_kpool, "is_hip", return_value=False),
            patch.object(dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool),
            patch.object(dsa_indexer_kpool, "deep_gemm", deep_gemm, create=True),
            patch(
                "sglang.kernels.ops.attention.dsa.tilelang_kernel."
                "tilelang_fp8_paged_mqa_logits",
                return_value=tilelang_logits,
            ) as tilelang_mqa,
        ):
            result = indexer._get_topk_paged(
                forward_batch, 7, q_fp8, weights, metadata
            )

        self.assertIs(result, topk_result)
        return (
            deep_gemm,
            indexer,
            pool_context_lens,
            pool_block_tables,
            stale_schedule,
            n_real,
            batch_size,
            next_n,
            num_heads,
            tilelang_mqa,
        )

    def test_target_verify_drops_deepep_padding_from_read_metadata(self):
        (
            _deep_gemm,
            indexer,
            pool_context_lens,
            pool_block_tables,
            _stale_schedule,
            n_real,
            _batch_size,
            _next_n,
            _num_heads,
            _tilelang_mqa,
        ) = self._run_paged_mqa_dispatch(ForwardMode.TARGET_VERIFY, metadata_rows=16)

        args = indexer._get_kpool_decode_metadata.call_args.args
        self.assertEqual(args[1].shape, (n_real, 2))
        self.assertEqual(args[2].shape, (n_real,))
        torch.testing.assert_close(
            args[1], torch.arange(32, dtype=torch.int32).view(16, 2)[:n_real]
        )
        torch.testing.assert_close(args[2], torch.arange(n_real, dtype=torch.int32))

    def test_target_verify_rebuilds_deepgemm_schedule_for_trimmed_token_layout(self):
        (
            deep_gemm,
            indexer,
            pool_context_lens,
            pool_block_tables,
            _stale_schedule,
            n_real,
            _batch_size,
            _next_n,
            num_heads,
            tilelang_mqa,
        ) = self._run_paged_mqa_dispatch(ForwardMode.TARGET_VERIFY, metadata_rows=16)

        tilelang_mqa.assert_not_called()
        deep_gemm.get_paged_mqa_logits_metadata.assert_called_once()
        schedule_call = deep_gemm.get_paged_mqa_logits_metadata.call_args
        torch.testing.assert_close(
            schedule_call.args[0], pool_context_lens.clamp(min=1)
        )
        self.assertEqual(schedule_call.args[1:], (64, 78))
        kernel_call = deep_gemm.fp8_paged_mqa_logits.call_args
        self.assertEqual(kernel_call.args[0].shape, (n_real, 1, num_heads, 128))
        torch.testing.assert_close(kernel_call.args[3], pool_context_lens)
        torch.testing.assert_close(kernel_call.args[4], pool_block_tables)
        self.assertIs(
            kernel_call.args[5],
            deep_gemm.get_paged_mqa_logits_metadata.return_value,
        )
        self.assertEqual(kernel_call.args[6], 128)
        self.assertFalse(kernel_call.kwargs["clean_logits"])
        self.assertFalse(
            indexer._get_kpool_decode_metadata.call_args.kwargs[
                "build_schedule_metadata"
            ]
        )

    def test_target_verify_reuses_deepgemm_schedule_without_metadata_padding(self):
        (
            deep_gemm,
            indexer,
            _pool_context_lens,
            _pool_block_tables,
            stale_schedule,
            _n_real,
            _batch_size,
            _next_n,
            _num_heads,
            _tilelang_mqa,
        ) = self._run_paged_mqa_dispatch(ForwardMode.TARGET_VERIFY)

        deep_gemm.get_paged_mqa_logits_metadata.assert_not_called()
        self.assertIs(deep_gemm.fp8_paged_mqa_logits.call_args.args[5], stale_schedule)
        self.assertTrue(
            indexer._get_kpool_decode_metadata.call_args.kwargs[
                "build_schedule_metadata"
            ]
        )

    def test_target_verify_zero_real_rows_returns_padding_sentinel(self):
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.index_kpool = 4
        indexer.index_topk = 8
        indexer._get_index_k_read_buffer = MagicMock()
        metadata = SimpleNamespace(
            get_page_table_64=lambda: torch.empty((0, 2), dtype=torch.int32),
            get_seqlens_expanded=lambda: torch.empty((0,), dtype=torch.int32),
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_token_non_padded_cpu=0,
        )
        pool = SimpleNamespace(page_size=64)

        with patch.object(
            dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool
        ):
            result = indexer._get_topk_paged(
                forward_batch,
                7,
                torch.empty((4, 32, 128), dtype=torch.float8_e4m3fn),
                torch.empty((4, 32, 1)),
                metadata,
            )

        self.assertEqual(result.shape, (4, 11))
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(torch.all(result == -1))
        indexer._get_index_k_read_buffer.assert_called_once_with(pool, 7)

    def test_draft_extend_v2_keeps_split_deepgemm_layout(self):
        (
            deep_gemm,
            indexer,
            pool_context_lens,
            pool_block_tables,
            stale_schedule,
            n_real,
            _batch_size,
            _next_n,
            num_heads,
            tilelang_mqa,
        ) = self._run_paged_mqa_dispatch(ForwardMode.DRAFT_EXTEND_V2)

        tilelang_mqa.assert_not_called()
        deep_gemm.get_paged_mqa_logits_metadata.assert_not_called()
        kernel_call = deep_gemm.fp8_paged_mqa_logits.call_args
        self.assertEqual(kernel_call.args[0].shape, (n_real, 1, num_heads, 128))
        torch.testing.assert_close(kernel_call.args[3], pool_context_lens)
        torch.testing.assert_close(kernel_call.args[4], pool_block_tables)
        self.assertIs(kernel_call.args[5], stale_schedule)
        self.assertTrue(
            indexer._get_kpool_decode_metadata.call_args.kwargs[
                "build_schedule_metadata"
            ]
        )

    def test_target_verify_cache_write_drops_mlp_sync_padding_tokens(self):
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.index_kpool_compress_ape = torch.empty(4, 128)
        indexer.scale_fmt = None
        indexer._get_q_k_bf16 = MagicMock(
            return_value=(
                torch.randn(8, 2, 128),
                torch.randn(8, 128),
                torch.randn(8, 128),
            )
        )
        indexer._compute_gate_score_if_missing = MagicMock(
            return_value=torch.randn(8, 128)
        )

        pool = MagicMock()
        pool.get_compress_tail_buffers.return_value = (
            torch.empty(1, 12, 128),
            torch.empty(1, 12, 128),
        )
        pool.get_index_k_with_scale_buffer.return_value = torch.empty(1)
        plan = MagicMock()
        plan.num_draft_tokens = 6
        plan.req = torch.tensor([17, 99])
        plan.write_start = torch.tensor([23, 0], dtype=torch.int32)
        plan.tail_logical_start = torch.tensor([19, 0], dtype=torch.int32)
        plan.write_loc = torch.tensor([[29], [-1]])
        plan.effective_n_per_batch = torch.tensor([6, 0], dtype=torch.int32)
        metadata = MagicMock()
        metadata.attn_metadata.kpool_write_plan = plan
        forward_batch = MagicMock()
        forward_batch.num_token_non_padded_cpu = 6
        forward_batch.spec_info.ragged_verify_layout = None
        forward_batch.out_cache_loc = torch.arange(8) + 100

        with (
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch.object(dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool),
            patch.object(kpool_fp8_index, "kpool_write_tail_and_maybe_compress") as write,
        ):
            result = indexer._forward_cuda_target_verify(
                x=torch.randn(8, 64),
                q_lora=torch.randn(8, 32),
                positions=torch.arange(8),
                forward_batch=forward_batch,
                layer_id=7,
                act_quant=MagicMock(),
                metadata=metadata,
                enable_dual_stream=False,
                return_indices=False,
            )

        self.assertIsNone(result)
        kwargs = write.call_args.kwargs
        self.assertEqual(kwargs["key"].shape, (6, 128))
        self.assertEqual(kwargs["score"].shape, (6, 128))
        torch.testing.assert_close(kwargs["out_cache_loc"], torch.arange(6) + 100)
        torch.testing.assert_close(kwargs["req_pool_indices"], torch.tensor([17]))
        torch.testing.assert_close(
            kwargs["write_start"], torch.tensor([23], dtype=torch.int32)
        )
        torch.testing.assert_close(
            kwargs["tail_logical_start"], torch.tensor([19], dtype=torch.int32)
        )
        torch.testing.assert_close(kwargs["write_loc"], torch.tensor([[29]]))
        torch.testing.assert_close(
            kwargs["effective_n_per_batch"],
            torch.tensor([6], dtype=torch.int32),
        )

    def test_target_verify_cache_write_rejects_ragged_groups(self):
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.index_kpool_compress_ape = torch.empty(4, 128)
        indexer.scale_fmt = None
        indexer._get_q_k_bf16 = MagicMock(
            return_value=(
                torch.randn(4, 2, 128),
                torch.randn(4, 128),
                torch.randn(4, 128),
            )
        )
        indexer._compute_gate_score_if_missing = MagicMock(
            return_value=torch.randn(4, 128)
        )
        pool = MagicMock()
        pool.get_compress_tail_buffers.return_value = (
            torch.empty(1, 12, 128),
            torch.empty(1, 12, 128),
        )
        pool.get_index_k_with_scale_buffer.return_value = torch.empty(1)
        plan = SimpleNamespace(num_draft_tokens=3)
        metadata = SimpleNamespace(
            attn_metadata=SimpleNamespace(kpool_write_plan=plan)
        )
        forward_batch = SimpleNamespace(
            num_token_non_padded_cpu=4,
            out_cache_loc=torch.arange(4),
            spec_info=SimpleNamespace(ragged_verify_layout=object()),
        )

        with (
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch.object(dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool),
            self.assertRaisesRegex(AssertionError, "fixed-width EAGLE groups"),
        ):
            indexer._forward_cuda_target_verify(
                x=torch.randn(4, 64),
                q_lora=torch.randn(4, 32),
                positions=torch.arange(4),
                forward_batch=forward_batch,
                layer_id=7,
                act_quant=MagicMock(),
                metadata=metadata,
                enable_dual_stream=False,
                return_indices=False,
            )
    def _make_decode_write_fixture(self, physical_batch, seq_lens):
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.index_kpool_compress_ape = torch.empty(4, 128)
        indexer.scale_fmt = None
        pool = MagicMock()
        metadata = MagicMock()
        metadata.get_seqlens_int32.return_value = torch.tensor(
            seq_lens, dtype=torch.int32
        )
        metadata.get_page_table_64.return_value = torch.arange(
            max(1, len(seq_lens)), dtype=torch.int32
        ).view(max(1, len(seq_lens)), 1)
        forward_batch = MagicMock()
        forward_batch.req_pool_indices = torch.arange(physical_batch)
        forward_batch.out_cache_loc = torch.arange(physical_batch) + 100
        key = torch.randn(physical_batch, 128)
        gate_score = torch.randn(physical_batch, 128)
        positions = torch.arange(physical_batch) + 16
        return indexer, pool, metadata, forward_batch, key, gate_score, positions

    def test_decode_cache_write_drops_mlp_sync_padding_rows(self):
        (
            indexer,
            pool,
            metadata,
            forward_batch,
            key,
            gate_score,
            positions,
        ) = self._make_decode_write_fixture(8, [17])

        with patch.object(dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool):
            indexer._compress_write_decode(
                key, gate_score, positions, forward_batch, 7, metadata
            )

        kwargs = pool.kpool_decode_update_index_cache.call_args.kwargs
        self.assertEqual(kwargs["key"].shape, (1, 128))
        self.assertEqual(kwargs["slot_score"].shape, (1, 128))
        torch.testing.assert_close(kwargs["positions"], torch.tensor([16]))
        torch.testing.assert_close(kwargs["req_pool_indices"], torch.tensor([0]))
        torch.testing.assert_close(kwargs["seq_lens"], torch.tensor([17], dtype=torch.int32))
        torch.testing.assert_close(kwargs["out_cache_loc"], torch.tensor([100]))

    def test_decode_cache_write_keeps_unpadded_rows(self):
        fixture = self._make_decode_write_fixture(2, [17, 23])
        indexer, pool, metadata, forward_batch, key, gate_score, positions = fixture

        with patch.object(dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool):
            indexer._compress_write_decode(
                key, gate_score, positions, forward_batch, 7, metadata
            )

        kwargs = pool.kpool_decode_update_index_cache.call_args.kwargs
        self.assertIs(kwargs["key"], key)
        self.assertIs(kwargs["slot_score"], gate_score)
        torch.testing.assert_close(kwargs["positions"], positions)

    def test_decode_cache_write_skips_empty_metadata(self):
        fixture = self._make_decode_write_fixture(8, [])
        indexer, pool, metadata, forward_batch, key, gate_score, positions = fixture

        with patch.object(dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool):
            indexer._compress_write_decode(
                key, gate_score, positions, forward_batch, 7, metadata
            )

        pool.kpool_decode_update_index_cache.assert_not_called()

    def test_decode_cache_write_rejects_metadata_rows_beyond_physical_rows(self):
        fixture = self._make_decode_write_fixture(1, [17, 23])
        indexer, pool, metadata, forward_batch, key, gate_score, positions = fixture

        with (
            patch.object(dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool),
            self.assertRaisesRegex(AssertionError, "more request rows than token rows"),
        ):
            indexer._compress_write_decode(
                key, gate_score, positions, forward_batch, 7, metadata
            )

        pool.kpool_decode_update_index_cache.assert_not_called()

    def test_cuda_tilelang_selector_reads_heads_from_unexpanded_query(self):
        with (
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            self.assertFalse(
                dsa_indexer_kpool.IndexerKPool._should_use_tilelang_paged_mqa_logits(
                    torch.empty(1, 32, 128)
                )
            )
            self.assertTrue(
                dsa_indexer_kpool.IndexerKPool._should_use_tilelang_paged_mqa_logits(
                    torch.empty(1, 16, 128)
                )
            )
    def test_rocm_uses_aiter_mqa_logits(self):
        marker = object()
        aiter_impl = MagicMock(return_value=marker)
        module = ModuleType("aiter.ops.triton.fp8_mqa_logits")
        module.fp8_mqa_logits = aiter_impl

        args = tuple(object() for _ in range(6))
        with (
            patch.object(dsa_indexer_kpool, "is_hip", return_value=True),
            patch.dict(
                sys.modules,
                {"aiter.ops.triton.fp8_mqa_logits": module},
            ),
        ):
            result = dsa_indexer_kpool.IndexerKPool._fp8_mqa_logits(
                *args, clean_logits=False
            )

        self.assertIs(result, marker)
        aiter_impl.assert_called_once_with(*args, clean_logits=False)

    def test_cuda_keeps_deep_gemm_mqa_logits(self):
        marker = object()
        deep_gemm = MagicMock()
        deep_gemm.fp8_mqa_logits.return_value = marker
        q_fp8, k_fp8, k_scale, weights, starts, ends = (object() for _ in range(6))

        with (
            patch.object(dsa_indexer_kpool, "is_hip", return_value=False),
            patch.object(dsa_indexer_kpool, "deep_gemm", deep_gemm, create=True),
        ):
            result = dsa_indexer_kpool.IndexerKPool._fp8_mqa_logits(
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                starts,
                ends,
                clean_logits=True,
            )

        self.assertIs(result, marker)
        deep_gemm.fp8_mqa_logits.assert_called_once_with(
            q_fp8,
            (k_fp8, k_scale),
            weights,
            starts,
            ends,
            clean_logits=True,
        )

    def test_portable_topk_masks_invalid_groups_and_expands(self):
        logits = torch.tensor([[0.1, 0.9, 0.8, 50.0]], dtype=torch.float32)
        result = _topk_from_pooled_history_logits_unfused(
            logits=logits,
            group_lengths=torch.tensor([3], dtype=torch.int32),
            pool_size=2,
            topk=4,
        )

        torch.testing.assert_close(
            result,
            torch.tensor([[2, 3, 4, 5]], dtype=torch.int32),
        )

    def test_rocm_uses_fused_kpool_topk_for_supported_group_count(self):
        logits = MagicMock()
        logits.ndim = 2
        logits.shape = (1, 512)
        logits.is_cuda = True
        logits.dtype = torch.float32
        group_lengths = torch.tensor([256], dtype=torch.int32)
        marker = MagicMock()
        marker.shape = (1, 2048)

        with (
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.kernels.ops.moe.kpool_topk_transform.fast_kpool_topk_transform_fused",
                return_value=marker,
            ) as fused,
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index._topk_from_pooled_history_logits_unfused"
            ) as unfused,
        ):
            result = topk_from_pooled_history_logits(
                logits=logits,
                group_lengths=group_lengths,
                pool_size=4,
                topk=2048,
            )

        self.assertIs(result, marker)
        fused.assert_called_once()
        unfused.assert_not_called()

    def test_rocm_keeps_2048_group_topk_on_unfused_path(self):
        logits = MagicMock()
        logits.ndim = 2
        logits.shape = (1, 2048)
        logits.is_cuda = True
        logits.dtype = torch.float32
        group_lengths = torch.tensor([2048], dtype=torch.int32)
        marker = object()

        with (
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index._topk_from_pooled_history_logits_unfused",
                return_value=marker,
            ) as unfused,
        ):
            result = topk_from_pooled_history_logits(
                logits=logits,
                group_lengths=group_lengths,
                pool_size=4,
                topk=8192,
            )

        self.assertIs(result, marker)
        unfused.assert_called_once()

    def test_cuda_keeps_supported_group_count_on_fused_path(self):
        logits = MagicMock()
        logits.ndim = 2
        logits.shape = (1, 512)
        logits.is_cuda = True
        logits.dtype = torch.float32
        group_lengths = torch.tensor([256], dtype=torch.int32)
        marker = MagicMock()
        marker.shape = (1, 2048)

        with (
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index.is_hip",
                return_value=False,
            ),
            patch(
                "sglang.kernels.ops.moe.kpool_topk_transform.fast_kpool_topk_transform_fused",
                return_value=marker,
            ) as fused,
        ):
            result = topk_from_pooled_history_logits(
                logits=logits,
                group_lengths=group_lengths,
                pool_size=4,
                topk=2048,
            )

        self.assertIs(result, marker)
        fused.assert_called_once()


if __name__ == "__main__":
    unittest.main()
