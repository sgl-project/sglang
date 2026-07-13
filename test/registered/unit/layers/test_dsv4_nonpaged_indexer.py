import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.indexer import FP8_DTYPE, C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import NonPagedIndexerPlan
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsv4.indexer"


class TestDSV4NonPagedIndexer(CustomTestCase):
    def _is_eligible(self, **overrides):
        backend = SimpleNamespace(hisparse_coordinator=None)
        c4_indexer = SimpleNamespace(use_fp4_indexer=overrides.get("fp4", False))
        forward_batch = SimpleNamespace(
            forward_mode=overrides.get("mode", ForwardMode.EXTEND),
            _original_forward_mode=overrides.get("original_mode"),
            tbo_parent_token_range=overrides.get("tbo"),
            batch_size=overrides.get("batch_size", 1),
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
            get_parallel().override(attn_cp_size=1),
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
        for case in (
            {"enabled": False},
            {"mode": ForwardMode.DECODE},
            {"original_mode": ForwardMode.DECODE},
            {"batch_size": 2},
            {"batch_size": 20_000},
            {"tbo": (1, 2)},
            {"prefill_graph": True},
            {"piecewise_graph": True},
            {"fp4": True},
        ):
            with self.subTest(case=case):
                self.assertFalse(self._is_eligible(**case))

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
        batch.extend_start_loc = torch.tensor([0, 2], dtype=torch.int32)
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


if __name__ == "__main__":
    unittest.main()
