"""CPU contract tests for DSA KPool logical request domains."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsaIndexerKPoolPadding(CustomTestCase):
    """Padding rows may join collectives, but never own KPool request state."""

    @staticmethod
    def _writer_indexer(num_tokens):
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.index_kpool_compress_ape = torch.empty(4, 128)
        indexer.scale_fmt = None
        indexer._get_q_k_bf16 = MagicMock(
            return_value=(
                torch.randn(num_tokens, 2, 128),
                torch.randn(num_tokens, 128),
                torch.randn(num_tokens, 128),
            )
        )
        indexer._compute_gate_score_if_missing = MagicMock(
            return_value=torch.randn(num_tokens, 128)
        )
        return indexer

    @staticmethod
    def _writer_pool():
        pool = MagicMock()
        pool.get_compress_tail_buffers.return_value = (
            torch.empty(1, 12, 128),
            torch.empty(1, 12, 128),
        )
        pool.get_index_k_with_scale_buffer.return_value = torch.empty(1)
        return pool

    @staticmethod
    def _write_plan(num_draft_tokens, rows=2):
        plan = SimpleNamespace(
            num_draft_tokens=num_draft_tokens,
            req=torch.tensor([17, 99]),
            write_start=torch.tensor([23, 0], dtype=torch.int32),
            tail_logical_start=torch.tensor([19, 0], dtype=torch.int32),
            write_loc=torch.tensor([[29], [-1]]),
            effective_n_per_batch=torch.tensor([6, 0], dtype=torch.int32),
        )
        for field in (
            "req",
            "write_start",
            "tail_logical_start",
            "write_loc",
            "effective_n_per_batch",
        ):
            setattr(plan, field, getattr(plan, field)[:rows])
        return plan

    def _run_verify(self, physical, real, num_draft_tokens, plan_rows=2, ragged=False):
        indexer = self._writer_indexer(physical)
        pool = self._writer_pool()
        plan = self._write_plan(num_draft_tokens, rows=plan_rows)
        metadata = SimpleNamespace(attn_metadata=SimpleNamespace(kpool_write_plan=plan))
        forward_batch = SimpleNamespace(
            global_num_token_non_padded_cpu=real,
            spec_info=SimpleNamespace(
                ragged_verify_layout=object() if ragged else None
            ),
            out_cache_loc=torch.arange(physical) + 100,
        )
        with (
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch.object(dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool),
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index."
                "kpool_write_tail_and_maybe_compress"
            ) as write,
        ):
            indexer._forward_cuda_target_verify(
                x=torch.randn(physical, 64),
                q_lora=torch.randn(physical, 32),
                positions=torch.arange(physical),
                forward_batch=forward_batch,
                layer_id=7,
                act_quant=MagicMock(),
                metadata=metadata,
                enable_dual_stream=False,
                return_indices=False,
            )
        return write

    def test_verify_write_trims_token_and_request_domains_together(self):
        write = self._run_verify(physical=8, real=6, num_draft_tokens=6)
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
            kwargs["effective_n_per_batch"], torch.tensor([6], dtype=torch.int32)
        )

    def test_verify_write_keeps_unpadded_token_and_request_domains(self):
        write = self._run_verify(physical=6, real=6, num_draft_tokens=3)
        kwargs = write.call_args.kwargs
        self.assertEqual(kwargs["key"].shape, (6, 128))
        torch.testing.assert_close(kwargs["req_pool_indices"], torch.tensor([17, 99]))
        torch.testing.assert_close(
            kwargs["write_start"], torch.tensor([23, 0], dtype=torch.int32)
        )
        torch.testing.assert_close(
            kwargs["tail_logical_start"], torch.tensor([19, 0], dtype=torch.int32)
        )
        torch.testing.assert_close(kwargs["write_loc"], torch.tensor([[29], [-1]]))
        torch.testing.assert_close(kwargs["out_cache_loc"], torch.arange(6) + 100)

    def test_verify_write_accepts_zero_real_tokens_without_request_state(self):
        write = self._run_verify(physical=4, real=0, num_draft_tokens=2)
        kwargs = write.call_args.kwargs
        self.assertEqual(kwargs["key"].shape, (0, 128))
        self.assertEqual(kwargs["score"].shape, (0, 128))
        self.assertEqual(kwargs["req_pool_indices"].shape, (0,))
        self.assertEqual(kwargs["write_start"].shape, (0,))
        self.assertEqual(kwargs["write_loc"].shape, (0, 1))
        self.assertEqual(kwargs["out_cache_loc"].shape, (0,))

    def test_verify_write_rejects_ragged_groups(self):
        with self.assertRaisesRegex(AssertionError, "fixed-width write plan"):
            self._run_verify(physical=4, real=4, num_draft_tokens=3, ragged=True)

    def test_verify_write_rejects_invalid_real_length(self):
        cases = (
            (8, 9, 2, "outside the physical input"),
            (5, 5, 3, "complete draft groups"),
        )
        for physical, real, draft, message in cases:
            with self.subTest(real=real, draft=draft):
                with self.assertRaisesRegex(AssertionError, message):
                    self._run_verify(physical, real, draft)

    def test_verify_write_rejects_real_requests_beyond_write_plan(self):
        with self.assertRaisesRegex(AssertionError, "outside the write plan"):
            self._run_verify(physical=6, real=6, num_draft_tokens=3, plan_rows=1)

    @staticmethod
    def _decode_write_fixture(physical, seq_lens):
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.index_kpool_compress_ape = torch.empty(4, 128)
        indexer.scale_fmt = None
        pool = MagicMock()
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        metadata = SimpleNamespace(
            get_seqlens_int32=lambda: seq_lens,
            get_page_table_64=lambda: torch.arange(
                seq_lens.numel() * 2, dtype=torch.int32
            ).view(seq_lens.numel(), 2),
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.arange(physical) + 10,
            out_cache_loc=torch.arange(physical) + 100,
        )
        return (
            indexer,
            pool,
            metadata,
            forward_batch,
            torch.randn(physical, 128),
            torch.randn(physical, 128),
            torch.arange(physical) + 16,
        )

    def _run_decode_write(self, physical, seq_lens):
        fixture = self._decode_write_fixture(physical, seq_lens)
        indexer, pool, metadata, forward_batch, key, score, positions = fixture
        with patch.object(dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool):
            indexer._compress_write_decode(
                key, score, positions, forward_batch, 7, metadata
            )
        return pool, key, score, positions

    def test_decode_write_trims_padded_rows_using_real_metadata(self):
        pool, _key, _score, positions = self._run_decode_write(8, [17])
        kwargs = pool.kpool_decode_update_index_cache.call_args.kwargs
        self.assertEqual(kwargs["key"].shape, (1, 128))
        self.assertEqual(kwargs["slot_score"].shape, (1, 128))
        torch.testing.assert_close(kwargs["positions"], positions[:1])
        torch.testing.assert_close(kwargs["req_pool_indices"], torch.tensor([10]))
        torch.testing.assert_close(
            kwargs["seq_lens"], torch.tensor([17], dtype=torch.int32)
        )
        torch.testing.assert_close(kwargs["out_cache_loc"], torch.tensor([100]))

    def test_decode_write_preserves_unpadded_request_rows(self):
        pool, key, score, positions = self._run_decode_write(2, [17, 23])
        kwargs = pool.kpool_decode_update_index_cache.call_args.kwargs
        torch.testing.assert_close(kwargs["key"], key)
        torch.testing.assert_close(kwargs["slot_score"], score)
        torch.testing.assert_close(kwargs["positions"], positions)
        torch.testing.assert_close(
            kwargs["seq_lens"], torch.tensor([17, 23], dtype=torch.int32)
        )

    def test_decode_write_skips_zero_real_requests(self):
        pool, _key, _score, _positions = self._run_decode_write(8, [])
        pool.kpool_decode_update_index_cache.assert_not_called()

    def test_decode_write_rejects_more_metadata_rows_than_physical_tokens(self):
        with self.assertRaisesRegex(
            AssertionError, "more request rows than token rows"
        ):
            self._run_decode_write(1, [17, 23])

    @staticmethod
    def _paged_fixture(
        mode,
        physical_rows,
        real_rows=None,
        num_heads=32,
        metadata_rows=None,
        plan_rows=None,
        use_plan=True,
    ):
        metadata_rows = physical_rows if metadata_rows is None else metadata_rows
        real_rows = physical_rows if real_rows is None else real_rows
        plan_rows = metadata_rows if plan_rows is None else plan_rows
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.index_kpool = 4
        indexer.index_topk = 8
        indexer.sm_count = 78
        indexer._get_index_k_read_buffer = MagicMock(
            return_value=torch.empty(2, 64 * 132, dtype=torch.uint8)
        )
        indexer._kpool_fused_topk_mapping = MagicMock(return_value=(None, None, None))
        result = object()
        indexer._topk_from_kpool_logits = MagicMock(return_value=result)
        pool_schedule = object() if use_plan else None
        pool_seqlens = (
            torch.arange(1, plan_rows + 1, dtype=torch.int32) if use_plan else None
        )
        plan = SimpleNamespace(
            pool_seqlens_per_q=pool_seqlens,
            pool_schedule_metadata=pool_schedule,
        )
        blocks = torch.arange(metadata_rows * 8, dtype=torch.int32).view(
            metadata_rows, 8
        )
        expanded = torch.arange(1, metadata_rows + 1, dtype=torch.int32)
        metadata = SimpleNamespace(
            attn_metadata=SimpleNamespace(
                kpool_write_plan=plan,
                pooled_cache_seqlens_int32=None,
                pooled_real_page_table=None,
                pooled_paged_mqa_schedule_metadata=None,
                pooled_index_kpool=None,
            ),
            get_page_table_64=lambda: blocks,
            get_seqlens_expanded=lambda: expanded,
            get_seqlens_int32=lambda: expanded,
        )
        forward_batch = SimpleNamespace(
            forward_mode=mode,
            global_num_token_non_padded_cpu=real_rows,
        )
        deep_gemm = MagicMock()
        rebuilt_schedule = object()
        deep_gemm.get_paged_mqa_logits_metadata.return_value = rebuilt_schedule
        deep_gemm.fp8_paged_mqa_logits.return_value = torch.empty(
            max(real_rows, 1), 128
        )
        return SimpleNamespace(
            indexer=indexer,
            metadata=metadata,
            forward_batch=forward_batch,
            q_fp8=torch.empty(physical_rows, num_heads, 128, dtype=torch.float8_e4m3fn),
            weights=torch.empty(physical_rows, num_heads, 1),
            pool=SimpleNamespace(page_size=64),
            deep_gemm=deep_gemm,
            stale_schedule=pool_schedule,
            rebuilt_schedule=rebuilt_schedule,
            plan=plan,
            blocks=blocks,
            result=result,
        )

    def _run_paged_reader(self, fixture, tilelang=False):
        if tilelang:
            selector = patch.object(
                dsa_indexer_kpool.IndexerKPool,
                "_should_use_tilelang_paged_mqa_logits",
                staticmethod(lambda _q_fp8: True),
            )
            tilelang_op = patch(
                "sglang.kernels.ops.attention.dsa.tilelang_kernel."
                "tilelang_fp8_paged_mqa_logits",
                return_value=torch.empty(
                    fixture.forward_batch.global_num_token_non_padded_cpu, 128
                ),
            )
        else:
            selector = nullcontext()
            tilelang_op = nullcontext(None)
        with (
            patch.object(
                dsa_indexer_kpool, "get_token_to_kv_pool", return_value=fixture.pool
            ),
            patch.object(
                dsa_indexer_kpool,
                "deep_gemm",
                fixture.deep_gemm,
                create=True,
            ),
            selector,
            tilelang_op as tilelang_mock,
        ):
            result = fixture.indexer._get_topk_paged(
                fixture.forward_batch,
                7,
                fixture.q_fp8,
                fixture.weights,
                fixture.metadata,
            )
        fixture.tilelang_mock = tilelang_mock
        return result

    def test_reader_trims_page_table_and_rebuilds_schedule_for_verify_prefix(self):
        fixture = self._paged_fixture(
            ForwardMode.TARGET_VERIFY, physical_rows=16, real_rows=6
        )
        self.assertIs(self._run_paged_reader(fixture), fixture.result)
        fixture.deep_gemm.get_paged_mqa_logits_metadata.assert_called_once()
        schedule_call = fixture.deep_gemm.get_paged_mqa_logits_metadata.call_args
        expected_context = fixture.plan.pool_seqlens_per_q[:6].view(-1, 1)
        torch.testing.assert_close(schedule_call.args[0], expected_context.clamp(min=1))
        self.assertEqual(schedule_call.args[1:], (64, 78))
        kernel_call = fixture.deep_gemm.fp8_paged_mqa_logits.call_args
        self.assertEqual(kernel_call.args[0].shape, (6, 1, 32, 128))
        torch.testing.assert_close(kernel_call.args[3], expected_context)
        torch.testing.assert_close(
            kernel_call.args[4], fixture.blocks[:6, ::4].contiguous()
        )
        self.assertIs(kernel_call.args[5], fixture.rebuilt_schedule)
        self.assertIsNot(kernel_call.args[5], fixture.stale_schedule)

    def test_reader_rebuilds_when_plan_has_more_rows_than_real_metadata(self):
        fixture = self._paged_fixture(
            ForwardMode.TARGET_VERIFY,
            physical_rows=6,
            real_rows=6,
            metadata_rows=6,
            plan_rows=16,
        )
        self.assertIs(self._run_paged_reader(fixture), fixture.result)
        fixture.deep_gemm.get_paged_mqa_logits_metadata.assert_called_once()
        expected_context = fixture.plan.pool_seqlens_per_q[:6].view(-1, 1)
        schedule_call = fixture.deep_gemm.get_paged_mqa_logits_metadata.call_args
        torch.testing.assert_close(schedule_call.args[0], expected_context.clamp(min=1))
        self.assertIs(fixture.plan.pool_schedule_metadata, fixture.stale_schedule)
        kernel_call = fixture.deep_gemm.fp8_paged_mqa_logits.call_args
        self.assertIs(kernel_call.args[5], fixture.rebuilt_schedule)

    def test_reader_reuses_existing_schedule_without_padding(self):
        fixture = self._paged_fixture(
            ForwardMode.TARGET_VERIFY, physical_rows=6, real_rows=6
        )
        self.assertIs(self._run_paged_reader(fixture), fixture.result)
        fixture.deep_gemm.get_paged_mqa_logits_metadata.assert_not_called()
        self.assertIs(
            fixture.deep_gemm.fp8_paged_mqa_logits.call_args.args[5],
            fixture.stale_schedule,
        )

    def test_reader_returns_sentinel_without_reading_zero_real_rows(self):
        fixture = self._paged_fixture(
            ForwardMode.TARGET_VERIFY, physical_rows=4, real_rows=0, metadata_rows=0
        )
        result = self._run_paged_reader(fixture)
        self.assertEqual(result.shape, (4, 11))
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(torch.all(result == -1))
        fixture.indexer._get_index_k_read_buffer.assert_not_called()

    def test_reader_rejects_real_rows_outside_metadata_domain(self):
        fixture = self._paged_fixture(
            ForwardMode.TARGET_VERIFY, physical_rows=2, real_rows=3
        )
        with self.assertRaisesRegex(AssertionError, "outside its metadata rows"):
            self._run_paged_reader(fixture)
        fixture.deep_gemm.get_paged_mqa_logits_metadata.assert_not_called()

    def test_reader_rejects_metadata_rows_beyond_physical_queries(self):
        fixture = self._paged_fixture(
            ForwardMode.TARGET_VERIFY,
            physical_rows=2,
            real_rows=3,
            metadata_rows=3,
        )
        with self.assertRaisesRegex(AssertionError, "more real rows than query rows"):
            self._run_paged_reader(fixture)
        fixture.indexer._get_index_k_read_buffer.assert_not_called()
        fixture.deep_gemm.get_paged_mqa_logits_metadata.assert_not_called()

    def test_draft_extend_builds_current_schedule_when_cached_metadata_is_absent(self):
        fixture = self._paged_fixture(
            ForwardMode.DRAFT_EXTEND_V2,
            physical_rows=6,
            real_rows=6,
            use_plan=False,
        )
        self.assertIs(self._run_paged_reader(fixture), fixture.result)
        fixture.deep_gemm.get_paged_mqa_logits_metadata.assert_called_once()
        expected_context = torch.div(
            torch.arange(1, 7, dtype=torch.int32), 4, rounding_mode="floor"
        ).view(-1, 1)
        schedule_call = fixture.deep_gemm.get_paged_mqa_logits_metadata.call_args
        torch.testing.assert_close(schedule_call.args[0], expected_context.clamp(min=1))
        kernel_call = fixture.deep_gemm.fp8_paged_mqa_logits.call_args
        self.assertEqual(kernel_call.args[0].shape, (6, 1, 32, 128))
        torch.testing.assert_close(kernel_call.args[3], expected_context)
        self.assertIs(kernel_call.args[5], fixture.rebuilt_schedule)

    def test_tilelang_reader_does_not_rebuild_deepgemm_schedule(self):
        fixture = self._paged_fixture(
            ForwardMode.TARGET_VERIFY,
            physical_rows=16,
            real_rows=6,
            num_heads=16,
        )
        self.assertIs(self._run_paged_reader(fixture, tilelang=True), fixture.result)
        fixture.deep_gemm.get_paged_mqa_logits_metadata.assert_not_called()
        fixture.deep_gemm.fp8_paged_mqa_logits.assert_not_called()
        fixture.tilelang_mock.assert_called_once()
        tilelang_call = fixture.tilelang_mock.call_args
        self.assertEqual(tilelang_call.args[0].shape, (6, 1, 16, 128))
        self.assertIsNone(tilelang_call.args[5])


if __name__ == "__main__":
    unittest.main()
