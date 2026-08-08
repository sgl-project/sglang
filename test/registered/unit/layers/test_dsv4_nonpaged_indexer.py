import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.indexer import FP8_DTYPE, C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import (
    NonPagedIndexerPlan,
    PagedIndexerMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsv4.indexer"


class TestDSV4PagedIndexerMetadata(CustomTestCase):
    def test_sm120_fp4_forces_deep_gemm_metadata(self):
        expected = torch.tensor([[0, 0], [1, 0]], dtype=torch.int32)
        deep_gemm = SimpleNamespace(
            get_num_sms=MagicMock(return_value=1),
            get_paged_mqa_logits_metadata=MagicMock(return_value=expected),
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
            envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(False),
            patch(
                "sglang.kernels.ops.attention.dsv4.get_paged_mqa_logits_metadata"
            ) as jit_metadata,
        ):
            metadata = PagedIndexerMetadata(
                page_size=256,
                page_table=torch.zeros((1, 1), dtype=torch.int32),
                c4_seq_lens=torch.tensor([65], dtype=torch.int32),
                force_deep_gemm_metadata=True,
            )

        self.assertIs(metadata.deep_gemm_metadata, expected)
        deep_gemm.get_num_sms.assert_called_once_with()
        deep_gemm.get_paged_mqa_logits_metadata.assert_called_once()
        args = deep_gemm.get_paged_mqa_logits_metadata.call_args.args
        torch.testing.assert_close(args[0], torch.tensor([[65]], dtype=torch.int32))
        self.assertEqual(args[1:], (64, 1))
        jit_metadata.assert_not_called()

    def test_sm120_fp8_torch_fallback_keeps_metadata_none(self):
        with (
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
            envs.SGLANG_OPT_USE_TOPK_V2.override(False),
        ):
            metadata = PagedIndexerMetadata(
                page_size=256,
                page_table=torch.zeros((1, 1), dtype=torch.int32),
                c4_seq_lens=torch.tensor([65], dtype=torch.int32),
            )

        self.assertIsNone(metadata.deep_gemm_metadata)


class TestDSV4NonPagedIndexer(CustomTestCase):
    def _is_eligible(self, **overrides):
        backend = SimpleNamespace(hisparse_coordinator=None)
        c4_indexer = SimpleNamespace(use_fp4_indexer=overrides.get("fp4", False))
        cp_size = overrides.get("cp_size", 1)
        forward_batch = SimpleNamespace(
            forward_mode=overrides.get("mode", ForwardMode.EXTEND),
            _original_forward_mode=overrides.get("original_mode"),
            tbo_parent_token_range=overrides.get("tbo"),
            batch_size=overrides.get("batch_size", 1),
            attn_cp_metadata=(
                SimpleNamespace()
                if cp_size > 1 and overrides.get("has_cp_metadata", True)
                else None
            ),
        )
        metadata = SimpleNamespace(
            use_prefill_cuda_graph=overrides.get("prefill_graph", False)
        )
        with (
            envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.override(
                overrides.get("enabled", True)
            ),
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            patch(f"{_INDEXER}.is_cuda", return_value=True),
            patch(f"{_INDEXER}.is_hip", return_value=False),
            get_parallel().override(
                attn_cp_size=cp_size,
                attn_cp_rank=overrides.get("cp_rank", 0),
            ),
            patch(
                f"{_INDEXER}.is_dsa_prefill_cp_round_robin_split",
                return_value=overrides.get("round_robin", True),
            ),
            patch(
                f"{_INDEXER}.is_in_tc_piecewise_cuda_graph",
                return_value=overrides.get("piecewise_graph", False),
            ),
            patch(f"{_INDEXER}.is_in_breakable_cuda_graph", return_value=False),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            return C4IndexerBackendMixin._can_use_nonpaged_indexer(
                backend,
                c4_indexer=c4_indexer,
                forward_batch=forward_batch,
                indexer_metadata=metadata,
            )

    def test_eligibility_is_fail_closed(self):
        self.assertIs(envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.default, True)
        self.assertEqual(
            envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS.default, 8192
        )
        self.assertTrue(self._is_eligible())
        self.assertTrue(self._is_eligible(batch_size=2))
        self.assertTrue(self._is_eligible(batch_size=20_000))
        self.assertTrue(self._is_eligible(cp_size=4, cp_rank=2))
        for case in (
            {"enabled": False},
            {"mode": ForwardMode.DECODE},
            {"original_mode": ForwardMode.DECODE},
            {"tbo": (1, 2)},
            {"prefill_graph": True},
            {"piecewise_graph": True},
            {"fp4": True},
            {"cp_size": 4, "round_robin": False},
            {"cp_size": 4, "has_cp_metadata": False},
        ):
            with self.subTest(case=case):
                self.assertFalse(self._is_eligible(**case))

    def test_round_robin_plan_uses_local_query_rows_and_k_prefix(self):
        backend = SimpleNamespace(_can_use_nonpaged_indexer=lambda **_: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)
        cp_size = 4
        global_query_rows = 15
        padded_local_query_rows = 4
        prefix_len = 6
        final_seq_len = prefix_len + global_query_rows
        threshold = envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS

        for cp_rank in range(cp_size):
            with self.subTest(cp_rank=cp_rank):
                local_causal_seq_lens = torch.arange(
                    prefix_len + cp_rank + 1,
                    final_seq_len + 1,
                    cp_size,
                    dtype=torch.int32,
                )
                logical_query_rows = local_causal_seq_lens.numel()
                c4_seq_lens = torch.div(local_causal_seq_lens, 4, rounding_mode="floor")
                c4_seq_lens = torch.nn.functional.pad(
                    c4_seq_lens,
                    (0, padded_local_query_rows - logical_query_rows),
                )
                page_table = torch.tensor([[3, 1]], dtype=torch.int32).repeat(
                    padded_local_query_rows, 1
                )
                batch = SimpleNamespace(
                    seq_lens=torch.tensor([final_seq_len], dtype=torch.int32),
                    seq_lens_cpu=[final_seq_len],
                    extend_seq_lens_cpu=[global_query_rows],
                    extend_seq_lens=torch.tensor(
                        [global_query_rows], dtype=torch.int32
                    ),
                    extend_start_loc=torch.tensor([0], dtype=torch.int32),
                    extend_num_tokens=global_query_rows,
                    attn_cp_metadata=SimpleNamespace(),
                )
                metadata = SimpleNamespace(nonpaged_plan=None, c4_page_size=64)

                with (
                    threshold.override(0),
                    get_parallel().override(attn_cp_size=cp_size, attn_cp_rank=cp_rank),
                ):
                    plan = C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                        backend,
                        c4_indexer=c4_indexer,
                        forward_batch=batch,
                        indexer_metadata=metadata,
                        page_table=page_table,
                        c4_seq_lens=c4_seq_lens,
                        query_rows=padded_local_query_rows,
                    )

                expected_gather_c4_len = final_seq_len // 4
                self.assertIsNotNone(plan)
                self.assertEqual(plan.query_rows, logical_query_rows)
                self.assertEqual(plan.seq_len_sum, expected_gather_c4_len)
                self.assertEqual(plan.max_seq_len, expected_gather_c4_len)
                self.assertEqual(plan.max_seqlen_k, 64)
                torch.testing.assert_close(plan.page_table, page_table[:1])
                torch.testing.assert_close(
                    plan.ke, c4_seq_lens[:logical_query_rows]
                )
                torch.testing.assert_close(
                    plan.gather_seq_lens,
                    torch.tensor([expected_gather_c4_len], dtype=torch.int32),
                )

    def test_multi_request_plan_concatenates_kv_and_builds_ragged_ranges(self):
        backend = SimpleNamespace(_can_use_nonpaged_indexer=lambda **_: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)
        extend_lens = [4, 3]
        query_rows = sum(extend_lens)
        batch = SimpleNamespace(
            seq_lens=torch.tensor([522, 260], dtype=torch.int32),
            seq_lens_cpu=[522, 260],
            extend_seq_lens_cpu=extend_lens,
            extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
            extend_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            extend_num_tokens=query_rows,
        )
        metadata = SimpleNamespace(nonpaged_plan=None, c4_page_size=64)
        request_0_table = torch.tensor([[3, 1, 4]], dtype=torch.int32).repeat(4, 1)
        request_1_table = torch.tensor([[9, 7, 8]], dtype=torch.int32).repeat(3, 1)
        page_table = torch.cat((request_0_table, request_1_table), dim=0)
        c4_seq_lens = torch.tensor(
            [127, 128, 129, 130, 63, 64, 65], dtype=torch.int32
        )

        threshold = envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS
        with (
            threshold.override(query_rows),
            get_parallel().override(attn_cp_size=1, attn_cp_rank=0),
        ):
            plan = C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                backend,
                c4_indexer=c4_indexer,
                forward_batch=batch,
                indexer_metadata=metadata,
                page_table=page_table,
                c4_seq_lens=c4_seq_lens,
                query_rows=query_rows,
            )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.query_rows, query_rows)
        self.assertEqual(plan.seq_len_sum, 195)
        self.assertEqual(plan.max_seq_len, 130)
        self.assertEqual(plan.max_seqlen_k, 192)
        torch.testing.assert_close(
            plan.page_table,
            torch.tensor([[3, 1, 4], [9, 7, 8]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            plan.gather_seq_lens,
            torch.tensor([130, 65], dtype=torch.int32),
        )
        torch.testing.assert_close(
            plan.ks,
            torch.tensor([0, 0, 0, 0, 130, 130, 130], dtype=torch.int32),
        )
        torch.testing.assert_close(
            plan.ke,
            torch.tensor([127, 128, 129, 130, 193, 194, 195], dtype=torch.int32),
        )

    def test_multi_request_round_robin_plan_selects_rank_local_requests(self):
        backend = SimpleNamespace(_can_use_nonpaged_indexer=lambda **_: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)
        extend_lens = [3, 5, 2]
        batch = SimpleNamespace(
            seq_lens=torch.tensor([11, 17, 10], dtype=torch.int32),
            seq_lens_cpu=[11, 17, 10],
            extend_seq_lens_cpu=extend_lens,
            extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
            extend_start_loc=torch.tensor([0, 3, 8], dtype=torch.int32),
            extend_num_tokens=sum(extend_lens),
            attn_cp_metadata=SimpleNamespace(),
        )
        metadata = SimpleNamespace(nonpaged_plan=None, c4_page_size=64)
        # Rank 2 owns flattened global query rows 2 and 6; the third row is
        # CP/DP padding and must not be sent to DeepGEMM or ragged top-k.
        page_table = torch.tensor([[3, 1], [9, 7], [0, 0]], dtype=torch.int32)
        c4_seq_lens = torch.tensor([2, 4, 123], dtype=torch.int32)

        threshold = envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS
        with (
            threshold.override(0),
            get_parallel().override(attn_cp_size=4, attn_cp_rank=2),
        ):
            plan = C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                backend,
                c4_indexer=c4_indexer,
                forward_batch=batch,
                indexer_metadata=metadata,
                page_table=page_table,
                c4_seq_lens=c4_seq_lens,
                query_rows=3,
            )

        self.assertIsNotNone(plan)
        torch.testing.assert_close(
            plan.page_table,
            torch.tensor([[3, 1], [9, 7]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            plan.gather_seq_lens,
            torch.tensor([2, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(plan.ks, torch.tensor([0, 2], dtype=torch.int32))
        torch.testing.assert_close(plan.ke, torch.tensor([2, 6], dtype=torch.int32))
        self.assertEqual(plan.query_rows, 2)
        self.assertEqual(plan.seq_len_sum, 6)
        self.assertEqual(plan.max_seq_len, 4)
        self.assertEqual(plan.max_seqlen_k, 64)

    def test_round_robin_query_threshold_uses_padded_rows_consistently(self):
        backend = SimpleNamespace(_can_use_nonpaged_indexer=lambda **_: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)
        metadata = SimpleNamespace(nonpaged_plan=None, c4_page_size=64)
        cp_size = 4

        def build_plan(global_query_rows, cp_rank):
            padded_local_query_rows = (global_query_rows + cp_size - 1) // cp_size
            local_causal_seq_lens = torch.arange(
                cp_rank + 1,
                global_query_rows + 1,
                cp_size,
                dtype=torch.int32,
            )
            c4_seq_lens = torch.div(local_causal_seq_lens, 4, rounding_mode="floor")
            c4_seq_lens = torch.nn.functional.pad(
                c4_seq_lens,
                (0, padded_local_query_rows - local_causal_seq_lens.numel()),
            )
            batch = SimpleNamespace(
                seq_lens=torch.tensor([global_query_rows], dtype=torch.int32),
                seq_lens_cpu=[global_query_rows],
                extend_seq_lens_cpu=[global_query_rows],
                extend_seq_lens=torch.tensor([global_query_rows], dtype=torch.int32),
                extend_start_loc=torch.tensor([0], dtype=torch.int32),
                extend_num_tokens=global_query_rows,
                attn_cp_metadata=SimpleNamespace(),
            )
            metadata.nonpaged_plan = None
            return C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                backend,
                c4_indexer=c4_indexer,
                forward_batch=batch,
                indexer_metadata=metadata,
                page_table=torch.zeros((padded_local_query_rows, 1), dtype=torch.int32),
                c4_seq_lens=c4_seq_lens,
                query_rows=padded_local_query_rows,
            )

        for cp_rank in range(cp_size):
            with (
                self.subTest(cp_rank=cp_rank),
                get_parallel().override(attn_cp_size=cp_size, attn_cp_rank=cp_rank),
            ):
                # All ranks have 8191 physical rows and therefore use paged.
                self.assertIsNone(build_plan(8191 * cp_size, cp_rank))

                # All ranks have 8192 padded physical rows. Rank 3 has only 8191
                # logical rows, but every rank must still choose non-paged.
                global_query_rows = 8192 * cp_size - 1
                plan = build_plan(global_query_rows, cp_rank)
                self.assertIsNotNone(plan)
                expected_logical_rows = len(
                    range(cp_rank + 1, global_query_rows + 1, cp_size)
                )
                self.assertEqual(plan.query_rows, expected_logical_rows)

    def test_single_request_plan_contract(self):
        backend = SimpleNamespace(_can_use_nonpaged_indexer=lambda **_: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)
        query_rows = 4
        batch = SimpleNamespace(
            seq_lens=torch.tensor([262], dtype=torch.int32),
            seq_lens_cpu=[262],
            extend_seq_lens_cpu=[query_rows],
            extend_seq_lens=torch.tensor([query_rows], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_num_tokens=query_rows,
        )
        metadata = SimpleNamespace(nonpaged_plan=None, c4_page_size=64)
        page_table = torch.tensor([[3, 1]], dtype=torch.int32).repeat(query_rows, 1)
        c4_seq_lens = torch.tensor([62, 63, 64, 65], dtype=torch.int32)

        def build_plan():
            with get_parallel().override(attn_cp_size=1, attn_cp_rank=0):
                return C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                    backend,
                    c4_indexer=c4_indexer,
                    forward_batch=batch,
                    indexer_metadata=metadata,
                    page_table=page_table,
                    c4_seq_lens=c4_seq_lens,
                    query_rows=query_rows,
                )

        threshold = envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS
        with threshold.override(threshold.default):
            self.assertIsNone(build_plan())
        with threshold.override(query_rows):
            plan = build_plan()
        self.assertEqual(
            (plan.seq_len_sum, plan.max_seqlen_k, plan.query_rows),
            (65, 128, query_rows),
        )
        torch.testing.assert_close(plan.page_table, page_table[:1])
        torch.testing.assert_close(plan.ke, c4_seq_lens)
        torch.testing.assert_close(plan.gather_seq_lens, c4_seq_lens[-1:])

        metadata.nonpaged_plan = None
        batch.extend_seq_lens_cpu = [2, 2]
        with threshold.override(0):
            self.assertIsNone(build_plan())

    def test_extreme_plan_metadata_is_bounded_and_fail_closed(self):
        backend = SimpleNamespace(_can_use_nonpaged_indexer=lambda **_: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)
        query_rows = 4
        batch = SimpleNamespace(
            seq_lens=torch.tensor([500_000], dtype=torch.int32),
            seq_lens_cpu=[500_000],
            extend_seq_lens_cpu=[query_rows],
            extend_seq_lens=torch.tensor([query_rows], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_num_tokens=query_rows,
        )
        metadata = SimpleNamespace(nonpaged_plan=None, c4_page_size=64)
        page_table = torch.zeros((query_rows, 1), dtype=torch.int32)
        c4_seq_lens = torch.tensor(
            [124_997, 124_998, 124_999, 125_000], dtype=torch.int32
        )

        def build_plan():
            with get_parallel().override(attn_cp_size=1, attn_cp_rank=0):
                return C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                    backend,
                    c4_indexer=c4_indexer,
                    forward_batch=batch,
                    indexer_metadata=metadata,
                    page_table=page_table,
                    c4_seq_lens=c4_seq_lens,
                    query_rows=query_rows,
                )

        threshold = envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS
        with threshold.override(query_rows):
            plan = build_plan()
        self.assertEqual(plan.seq_len_sum, 125_000)
        self.assertEqual(plan.max_seq_len, 125_000)
        self.assertEqual(plan.max_seqlen_k, 125_056)

        metadata.nonpaged_plan = None
        batch.seq_lens = torch.tensor([500_000, 200], dtype=torch.int32)
        batch.seq_lens_cpu = [500_000, 200]
        batch.extend_seq_lens_cpu = [2, 2]
        batch.extend_seq_lens = torch.tensor([2, 2], dtype=torch.int32)
        batch.extend_start_loc = torch.tensor([0], dtype=torch.int32)
        with threshold.override(query_rows):
            self.assertIsNone(build_plan())

    def test_query_threshold_boundary(self):
        can_use_nonpaged_indexer = MagicMock(return_value=True)
        backend = SimpleNamespace(_can_use_nonpaged_indexer=can_use_nonpaged_indexer)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)
        metadata = SimpleNamespace(nonpaged_plan=None, c4_page_size=64)

        def build_plan(query_rows):
            batch = SimpleNamespace(
                seq_lens=torch.tensor([query_rows], dtype=torch.int32),
                seq_lens_cpu=[query_rows],
                extend_seq_lens_cpu=[query_rows],
                extend_seq_lens=torch.tensor([query_rows], dtype=torch.int32),
                extend_start_loc=torch.tensor([0], dtype=torch.int32),
                extend_num_tokens=query_rows,
            )
            c4_seq_lens = torch.div(
                torch.arange(1, query_rows + 1, dtype=torch.int32),
                4,
                rounding_mode="floor",
            ).clamp_min_(1)
            with get_parallel().override(attn_cp_size=1, attn_cp_rank=0):
                return C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                    backend,
                    c4_indexer=c4_indexer,
                    forward_batch=batch,
                    indexer_metadata=metadata,
                    page_table=torch.zeros((query_rows, 1), dtype=torch.int32),
                    c4_seq_lens=c4_seq_lens,
                    query_rows=query_rows,
                )

        for query_rows, expected in ((8191, False), (8192, True), (8193, True)):
            with self.subTest(query_rows=query_rows):
                metadata.nonpaged_plan = None
                can_use_nonpaged_indexer.reset_mock()
                self.assertIs(build_plan(query_rows) is not None, expected)
                if expected:
                    can_use_nonpaged_indexer.assert_called_once()
                else:
                    can_use_nonpaged_indexer.assert_not_called()

        metadata.nonpaged_plan = None
        threshold = envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS
        with threshold.override(8193):
            self.assertIsNone(build_plan(8192))

    def test_nonpaged_dispatch_uses_gathered_kv_contract(self):
        query_rows = 4
        plan = NonPagedIndexerPlan(
            page_table=torch.tensor([[3, 1]], dtype=torch.int32),
            gather_seq_lens=torch.tensor([65], dtype=torch.int32),
            ks=torch.zeros(query_rows, dtype=torch.int32),
            ke=torch.tensor([62, 63, 64, 65], dtype=torch.int32),
            seq_len_sum=65,
            max_seq_len=65,
            max_seqlen_k=128,
            query_rows=query_rows,
        )
        q_indexer = torch.zeros((6, 2, 128), dtype=torch.uint8).view(FP8_DTYPE)
        weights = torch.ones((6, 2), dtype=torch.float32)
        k_u8 = torch.zeros((65, 128), dtype=torch.uint8)
        scale_u8 = torch.zeros((65, 4), dtype=torch.uint8)
        token_to_kv_pool = MagicMock()
        token_to_kv_pool.get_index_k_scale_buffer.return_value = (k_u8, scale_u8)
        c4_indexer = SimpleNamespace(layer_id=17)
        expected = MagicMock(name="logits")
        deep_gemm = SimpleNamespace(fp8_mqa_logits=MagicMock(return_value=expected))

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            actual = C4IndexerBackendMixin._forward_nonpaged_indexer(
                q_indexer=q_indexer,
                weights=weights,
                c4_indexer=c4_indexer,
                token_to_kv_pool=token_to_kv_pool,
                plan=plan,
            )

        self.assertIs(actual, expected)
        token_to_kv_pool.get_index_k_scale_buffer.assert_called_once_with(
            layer_id=17,
            seq_len_tensor=plan.gather_seq_lens,
            page_indices=plan.page_table,
            seq_len_sum=65,
            max_seq_len=65,
        )
        call = deep_gemm.fp8_mqa_logits.call_args
        torch.testing.assert_close(call.args[0], q_indexer[:query_rows])
        torch.testing.assert_close(call.args[1][0], k_u8.view(FP8_DTYPE))
        torch.testing.assert_close(
            call.args[1][1], scale_u8.view(torch.float32).squeeze(-1)
        )
        torch.testing.assert_close(call.args[2], weights[:query_rows])
        torch.testing.assert_close(call.args[3], plan.ks)
        torch.testing.assert_close(call.args[4], plan.ke)
        self.assertEqual(call.kwargs, {"clean_logits": False, "max_seqlen_k": 128})

    def test_topk_v2_plan_rebuilt_once_after_cp_padding_is_trimmed(self):
        physical_query_rows = 3
        logical_query_rows = 2
        page_table = torch.tensor([[3], [3], [0]], dtype=torch.int32)
        physical_c4_seq_lens = torch.tensor([9, 11, 0], dtype=torch.int32)
        initial_topk_plan = torch.tensor(
            [[100, 3], [0, 9], [1, 11], [2, 0]], dtype=torch.int32
        )
        rebuilt_topk_plan = torch.tensor(
            [[200, 2], [0, 9], [1, 11]], dtype=torch.int32
        )

        nonpaged_plan = NonPagedIndexerPlan(
            page_table=page_table[:1],
            gather_seq_lens=torch.tensor([11], dtype=torch.int32),
            ks=torch.zeros(logical_query_rows, dtype=torch.int32),
            ke=physical_c4_seq_lens[:logical_query_rows],
            seq_len_sum=11,
            max_seq_len=11,
            max_seqlen_k=64,
            query_rows=logical_query_rows,
        )
        q_indexer = torch.zeros(
            (physical_query_rows, 2, 128), dtype=torch.float32
        )
        weights = torch.ones((physical_query_rows, 2, 1), dtype=torch.float32)
        logits = torch.zeros((logical_query_rows, 16), dtype=torch.float32)
        c4_sparse_page_indices = torch.full(
            (physical_query_rows, 512), -1, dtype=torch.int32
        )
        core_metadata = SimpleNamespace(
            positions=torch.arange(physical_query_rows),
            page_table=page_table,
            c4_sparse_page_indices=c4_sparse_page_indices,
            c4_sparse_raw_indices=None,
        )
        backend = SimpleNamespace(
            token_to_kv_pool=MagicMock(),
            forward_metadata=None,
            _forward_prepare_normal=MagicMock(return_value=(q_indexer, weights)),
            _get_nonpaged_indexer_plan=MagicMock(return_value=nonpaged_plan),
            _forward_nonpaged_indexer=MagicMock(return_value=logits),
            debug_use_external_c4_sparse_indices=False,
            hisparse_coordinator=None,
            dsa_topk_backend=SimpleNamespace(
                is_torch=MagicMock(return_value=False),
                is_flashinfer=MagicMock(return_value=False),
            ),
        )
        c4_indexer = SimpleNamespace(use_fp4_indexer=False, layer_id=17)
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        deep_gemm = SimpleNamespace(
            get_num_sms=MagicMock(return_value=1),
            get_paged_mqa_logits_metadata=MagicMock(
                return_value=torch.zeros(
                    (physical_query_rows, 2), dtype=torch.int32
                )
            ),
            fp8_paged_mqa_logits=MagicMock(),
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
            envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.override(False),
            envs.SGLANG_OPT_USE_TOPK_V2.override(True),
            envs.SGLANG_TOPK_TRANSFORM_512_TORCH.override(False),
            patch(
                "sglang.kernels.ops.attention.dsv4.plan_topk_v2",
                return_value=initial_topk_plan,
            ) as initial_plan,
            patch(f"{_INDEXER}.plan_topk_v2", return_value=rebuilt_topk_plan) as replan,
            patch(f"{_INDEXER}.topk_transform_512_v2") as transform,
            patch(f"{_INDEXER}.get_global_indexer_capturer", return_value=None),
        ):
            indexer_metadata = PagedIndexerMetadata(
                page_size=256,
                page_table=page_table,
                c4_seq_lens=physical_c4_seq_lens,
            )
            backend.forward_metadata = SimpleNamespace(
                indexer_metadata=indexer_metadata,
                core_metadata=core_metadata,
            )

            for _ in range(2):
                C4IndexerBackendMixin.forward_c4_indexer(
                    backend,
                    x=torch.zeros((physical_query_rows, 1)),
                    q_lora=torch.zeros((physical_query_rows, 1)),
                    c4_indexer=c4_indexer,
                    forward_batch=forward_batch,
                )

        initial_plan.assert_called_once_with(physical_c4_seq_lens)
        replan.assert_called_once()
        torch.testing.assert_close(
            replan.call_args.args[0], physical_c4_seq_lens[:logical_query_rows]
        )
        self.assertIs(indexer_metadata.topk_metadata, rebuilt_topk_plan)
        self.assertEqual(transform.call_count, 2)
        for call in transform.call_args_list:
            torch.testing.assert_close(
                call.args[1], physical_c4_seq_lens[:logical_query_rows]
            )
            self.assertIs(call.args[5], rebuilt_topk_plan)


if __name__ == "__main__":
    unittest.main()
