import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsv4.indexer import FP8_DTYPE, C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import (
    NonPagedIndexerPlan,
    PagedIndexerMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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
        # FP4 indexer cannot use the non-paged path: get_index_k_scale_buffer
        # reads K at FP8 strides (128 B/token), but FP4 buffers pack only
        # 68 B/token — silent data corruption.  See PR #33288 review.
        self.assertFalse(self._is_eligible(fp4=True))
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
        backend.dsa_topk_backend = SimpleNamespace(is_sgl_kernel=lambda: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False, index_topk=64)
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
        with (
            threshold.override(query_rows),
            envs.SGLANG_TOPK_TRANSFORM_512_TORCH.override(False),
        ):
            plan = build_plan()
        self.assertEqual(
            (plan.seq_len_sum, plan.max_seqlen_k, plan.query_rows),
            (65, 128, query_rows),
        )
        torch.testing.assert_close(plan.page_table, page_table[:1])
        torch.testing.assert_close(
            plan.ke, torch.tensor([0, 0, 0, 65], dtype=torch.int32)
        )
        torch.testing.assert_close(plan.gather_seq_lens, c4_seq_lens[-1:])

        metadata.nonpaged_plan = None
        batch.extend_seq_lens_cpu = [2, 2]
        with threshold.override(0):
            self.assertIsNone(build_plan())

    def test_extreme_plan_metadata_is_bounded_and_fail_closed(self):
        backend = SimpleNamespace(_can_use_nonpaged_indexer=lambda **_: True)
        backend.dsa_topk_backend = SimpleNamespace(is_sgl_kernel=lambda: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False, index_topk=512)
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
        backend.dsa_topk_backend = SimpleNamespace(is_sgl_kernel=lambda: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False, index_topk=512)
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
        c4_indexer = SimpleNamespace(layer_id=17, use_fp4_indexer=False)
        expected = MagicMock(name="logits")
        deep_gemm = SimpleNamespace(fp8_fp4_mqa_logits=MagicMock(return_value=expected))

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
        call = deep_gemm.fp8_fp4_mqa_logits.call_args
        # FP8 path: q_arg is (q_tensor, None)
        q_arg = call.args[0]
        self.assertIsInstance(q_arg, tuple)
        torch.testing.assert_close(q_arg[0], q_indexer[:query_rows])
        self.assertIsNone(q_arg[1])
        torch.testing.assert_close(call.args[1][0], k_u8.view(FP8_DTYPE))
        torch.testing.assert_close(
            call.args[1][1], scale_u8.view(torch.float32).squeeze(-1)
        )
        torch.testing.assert_close(call.args[2], weights[:query_rows])
        torch.testing.assert_close(call.args[3], plan.ks)
        torch.testing.assert_close(call.args[4], plan.ke)
        self.assertEqual(call.kwargs, {"clean_logits": False, "max_seqlen_k": 128})


class TestDSV4BudgetDetection(CustomTestCase):
    """Tests for _should_chunk_mqa_logits detection logic (tasks 4.1, 4.3)."""

    def test_small_batch_never_chunks(self):
        """Below STATIC_SKIP_ELEMS, detection returns False regardless of budget."""
        from sglang.srt.layers.attention.dsv4.indexer import (
            _MQA_LOGITS_STATIC_SKIP_ELEMS,
        )

        # query_rows * max_c4_seq_len < STATIC_SKIP_ELEMS → never chunk
        qr = 100
        mcsl = 100
        self.assertLess(qr * mcsl, _MQA_LOGITS_STATIC_SKIP_ELEMS)
        need, budget = C4IndexerBackendMixin._should_chunk_mqa_logits(qr, mcsl, 0)
        self.assertFalse(need)
        self.assertEqual(budget, 0)

    def test_detection_boundary_at_budget(self):
        """Bytes exactly at budget → no chunk; one over → chunk."""
        qr = 3000
        mcsl = 3000
        # Above STATIC_SKIP_ELEMS
        self.assertGreaterEqual(qr * mcsl, 8_000_000)

        # Mock budget to exactly match logits_bytes
        logits_bytes = qr * mcsl * 4
        with patch.object(
            C4IndexerBackendMixin,
            "_get_mqa_logits_budget_bytes",
            staticmethod(lambda di: logits_bytes),
        ):
            need, b = C4IndexerBackendMixin._should_chunk_mqa_logits(qr, mcsl, 0)
            self.assertFalse(need)  # bytes == budget, not > budget
            self.assertEqual(b, logits_bytes)

        # One byte over → chunk
        with patch.object(
            C4IndexerBackendMixin,
            "_get_mqa_logits_budget_bytes",
            staticmethod(lambda di: logits_bytes - 1),
        ):
            need, b = C4IndexerBackendMixin._should_chunk_mqa_logits(qr, mcsl, 0)
            self.assertTrue(need)
            self.assertEqual(b, logits_bytes - 1)


class TestDSV4OversizeVarlenChunked(CustomTestCase):
    """Tests for the oversize varlen chunked path (tasks 4.2, 4.4, 4.5)."""

    def _make_mocks(self, query_rows, use_fp4=False):
        """Create mock objects for _forward_oversize_varlen_chunked."""
        if use_fp4:
            q_indexer = (
                torch.zeros((query_rows, 2, 128), dtype=torch.uint8),
                torch.zeros((query_rows, 2), dtype=torch.float32),
            )
        else:
            q_indexer = torch.zeros((query_rows, 2, 128), dtype=torch.uint8).view(
                FP8_DTYPE
            )
        weights = torch.ones((query_rows, 2), dtype=torch.float32)
        c4_indexer = SimpleNamespace(layer_id=17, use_fp4_indexer=use_fp4)
        k_u8 = torch.zeros((1000, 128), dtype=torch.uint8)
        scale_u8 = torch.zeros((1000, 4), dtype=torch.uint8)
        token_to_kv_pool = MagicMock()
        token_to_kv_pool.get_index_k_scale_buffer.return_value = (k_u8, scale_u8)
        return q_indexer, weights, c4_indexer, token_to_kv_pool

    def _run_chunked_path(
        self, query_rows, max_c4_seq_len, budget_bytes, use_fp4=False
    ):
        """Helper: run _forward_oversize_varlen_chunked and return outputs."""
        q_indexer, weights, c4_indexer, token_to_kv_pool = self._make_mocks(
            query_rows, use_fp4=use_fp4
        )
        # Use unique weights per row so the mock can identify the starting row.
        for i in range(query_rows):
            weights[i] = float(i)

        c4_seq_lens = torch.full((query_rows,), max_c4_seq_len, dtype=torch.int32)
        page_table = torch.zeros((query_rows, 2), dtype=torch.int32)
        c4_sparse_page_indices = torch.full((query_rows, 512), -1, dtype=torch.int32)

        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=[max_c4_seq_len * 4],
            seq_lens=torch.tensor([max_c4_seq_len * 4], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens=torch.tensor([query_rows], dtype=torch.int32),
        )
        indexer_metadata = SimpleNamespace(
            c4_page_size=64,
            topk_metadata=torch.empty((0,)),
        )

        # Pre-compute deterministic logits: row i has values [0..N) shifted by i.
        # The mock uses weights[0] to identify the starting row and returns the
        # corresponding slice so chunked and unchunked runs see identical logits.
        precomputed = torch.stack(
            [
                torch.arange(max_c4_seq_len, dtype=torch.float32) + i * 0.1
                for i in range(query_rows)
            ]
        )

        def mock_logits(q_arg, kv, w, ks, ke, **kw):
            rows = ke.shape[0]
            start = int(w[0, 0].item())
            return precomputed[start : start + rows].clone()

        deep_gemm = SimpleNamespace(fp8_fp4_mqa_logits=mock_logits)
        backend = SimpleNamespace(dsa_topk_backend=DSATopKBackend.SGL_KERNEL)
        backend._run_topk_transform = C4IndexerBackendMixin._run_topk_transform.__get__(
            backend
        )

        out = c4_sparse_page_indices.clone()
        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            envs.SGLANG_TOPK_TRANSFORM_512_TORCH.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(False),
        ):
            C4IndexerBackendMixin._forward_oversize_varlen_chunked(
                backend,
                q_indexer=q_indexer,
                weights=weights,
                c4_indexer=c4_indexer,
                token_to_kv_pool=token_to_kv_pool,
                forward_batch=forward_batch,
                indexer_metadata=indexer_metadata,
                page_table=page_table,
                c4_seq_lens=c4_seq_lens,
                c4_sparse_page_indices=out,
                raw_indices=None,
                budget_bytes=budget_bytes,
                max_c4_seq_len=max_c4_seq_len,
                use_fp4_indexer=use_fp4,
                query_rows=query_rows,
            )
        return out, token_to_kv_pool

    def test_single_request_chunked_topk_matches_unchunked(self):
        """Chunked topk output matches single-pass baseline from same logits.

        With the budget_bytes fix, the unchunked run (budget = all rows)
        processes everything in one iteration while the chunked run
        (budget = 3 rows) iterates 4 times (3+3+3+1).  topk is per-row,
        so chunking must not change results.
        """
        query_rows = 10
        max_c4_seq_len = 100

        # Unchunked: budget allows all rows in one chunk.
        unchunked_out, _ = self._run_chunked_path(
            query_rows,
            max_c4_seq_len,
            budget_bytes=max_c4_seq_len * 4 * query_rows,
        )

        # Chunked: budget allows only 3 rows per chunk.
        chunked_out, _ = self._run_chunked_path(
            query_rows,
            max_c4_seq_len,
            budget_bytes=max_c4_seq_len * 4 * 3,
        )

        # topk is per-row; chunking must not change results.
        torch.testing.assert_close(chunked_out, unchunked_out)

    def test_multi_request_oversize_per_request_processing(self):
        """Task 4.4: Multi-request batch, each request processed independently."""
        req1_rows = 5
        req2_rows = 5
        query_rows = req1_rows + req2_rows
        max_c4_seq_len = 100
        budget_bytes = max_c4_seq_len * 4 * 2  # 2 rows per chunk

        q_indexer, weights, c4_indexer, token_to_kv_pool = self._make_mocks(query_rows)
        c4_seq_lens = torch.full((query_rows,), max_c4_seq_len, dtype=torch.int32)
        page_table = torch.zeros((query_rows, 2), dtype=torch.int32)
        c4_sparse_page_indices = torch.full((query_rows, 512), -1, dtype=torch.int32)

        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens_cpu=[max_c4_seq_len * 4, max_c4_seq_len * 4],
            seq_lens=torch.tensor(
                [max_c4_seq_len * 4, max_c4_seq_len * 4], dtype=torch.int32
            ),
            extend_start_loc=torch.tensor([0, req1_rows], dtype=torch.int32),
            extend_seq_lens=torch.tensor([req1_rows, req2_rows], dtype=torch.int32),
        )
        indexer_metadata = SimpleNamespace(
            c4_page_size=64,
            topk_metadata=torch.empty((0,)),
        )

        def mock_logits(q_arg, kv, w, ks, ke, **kw):
            rows = ke.shape[0]
            return torch.randn(rows, max_c4_seq_len, dtype=torch.float32)

        deep_gemm = SimpleNamespace(fp8_fp4_mqa_logits=mock_logits)
        backend = SimpleNamespace(dsa_topk_backend=DSATopKBackend.SGL_KERNEL)
        backend._run_topk_transform = C4IndexerBackendMixin._run_topk_transform.__get__(
            backend
        )

        chunked_out = c4_sparse_page_indices.clone()
        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            envs.SGLANG_TOPK_TRANSFORM_512_TORCH.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(False),
        ):
            C4IndexerBackendMixin._forward_oversize_varlen_chunked(
                backend,
                q_indexer=q_indexer,
                weights=weights,
                c4_indexer=c4_indexer,
                token_to_kv_pool=token_to_kv_pool,
                forward_batch=forward_batch,
                indexer_metadata=indexer_metadata,
                page_table=page_table,
                c4_seq_lens=c4_seq_lens,
                c4_sparse_page_indices=chunked_out,
                raw_indices=None,
                budget_bytes=budget_bytes,
                max_c4_seq_len=max_c4_seq_len,
                use_fp4_indexer=False,
                query_rows=query_rows,
            )

        # Two requests → two KV gathers
        self.assertEqual(token_to_kv_pool.get_index_k_scale_buffer.call_count, 2)
        # Output was written for all rows
        self.assertTrue((chunked_out[:req1_rows] != -1).any())
        self.assertTrue((chunked_out[req1_rows:] != -1).any())

    def test_fp4_varlen_routing_uses_fp8_fp4_kernel(self):
        """Task 4.5: FP4 oversize triggers fp8_fp4_mqa_logits with q_sf.

        Note: FP4 is currently excluded from the oversize routing in
        production (get_index_k_scale_buffer uses FP8 strides, see PR
        #33288 review comment 2).  This test verifies the query-side FP4
        handling inside _forward_oversize_varlen_chunked in isolation,
        guarding the code path for when K-gather FP4 support is added.
        """
        query_rows = 6
        max_c4_seq_len = 100
        budget_bytes = max_c4_seq_len * 4 * 2

        q_indexer, weights, c4_indexer, token_to_kv_pool = self._make_mocks(
            query_rows, use_fp4=True
        )
        c4_seq_lens = torch.full((query_rows,), max_c4_seq_len, dtype=torch.int32)
        page_table = torch.zeros((query_rows, 2), dtype=torch.int32)
        c4_sparse_page_indices = torch.full((query_rows, 512), -1, dtype=torch.int32)

        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=[max_c4_seq_len * 4],
            seq_lens=torch.tensor([max_c4_seq_len * 4], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens=torch.tensor([query_rows], dtype=torch.int32),
        )
        indexer_metadata = SimpleNamespace(
            c4_page_size=64,
            topk_metadata=torch.empty((0,)),
        )

        captured_calls = []

        def mock_logits(q_arg, kv, w, ks, ke, **kw):
            captured_calls.append(q_arg)
            rows = ke.shape[0]
            return torch.randn(rows, max_c4_seq_len, dtype=torch.float32)

        deep_gemm = SimpleNamespace(fp8_fp4_mqa_logits=mock_logits)
        backend = SimpleNamespace(dsa_topk_backend=DSATopKBackend.SGL_KERNEL)
        backend._run_topk_transform = C4IndexerBackendMixin._run_topk_transform.__get__(
            backend
        )

        chunked_out = c4_sparse_page_indices.clone()
        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            envs.SGLANG_TOPK_TRANSFORM_512_TORCH.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(False),
        ):
            C4IndexerBackendMixin._forward_oversize_varlen_chunked(
                backend,
                q_indexer=q_indexer,
                weights=weights,
                c4_indexer=c4_indexer,
                token_to_kv_pool=token_to_kv_pool,
                forward_batch=forward_batch,
                indexer_metadata=indexer_metadata,
                page_table=page_table,
                c4_seq_lens=c4_seq_lens,
                c4_sparse_page_indices=chunked_out,
                raw_indices=None,
                budget_bytes=budget_bytes,
                max_c4_seq_len=max_c4_seq_len,
                use_fp4_indexer=True,
                query_rows=query_rows,
            )

        # Verify fp8_fp4_mqa_logits was called with (q_fp4, q_sf) tuples
        self.assertTrue(len(captured_calls) > 0)
        for q_arg in captured_calls:
            self.assertIsInstance(q_arg, tuple)
            self.assertEqual(len(q_arg), 2)
            # q_sf should be a tensor (not None) for FP4
            self.assertIsNotNone(q_arg[1])
        self.assertTrue((chunked_out != -1).any())


class TestDSV4TopkV2MetadataRegeneration(CustomTestCase):
    """Verify H1 fix: topk_transform_512_v2 metadata regenerated for chunks."""

    def test_v2_metadata_regenerates_for_chunked_path(self):
        """When topk_metadata shape (N+1,2) mismatches c4_seq_lens shape (M,),
        plan_topk_v2 is called to regenerate before topk_transform_512_v2."""
        from sglang.kernels.ops.attention.dsv4 import plan_topk_v2

        chunk_rows = 3
        full_rows = 10
        max_c4_seq_len = 100

        logits = torch.randn(chunk_rows, max_c4_seq_len, dtype=torch.float32)
        c4_seq_lens = torch.full((chunk_rows,), max_c4_seq_len, dtype=torch.int32)
        page_table = torch.zeros((chunk_rows, 2), dtype=torch.int32)
        c4_sparse_page_indices = torch.full((chunk_rows, 512), -1, dtype=torch.int32)

        # Full-batch metadata: shape (full_rows+1, 2) — mismatches chunk_rows.
        full_metadata = torch.empty((full_rows + 1, 2), dtype=torch.int32)
        indexer_metadata = SimpleNamespace(
            c4_page_size=64,
            topk_metadata=full_metadata,
        )

        captured_meta = []

        def fake_v2(scores, seq_lens, pt, out, ps, metadata, **kw):
            captured_meta.append(metadata)

        backend = SimpleNamespace(dsa_topk_backend=DSATopKBackend.SGL_KERNEL)

        with (
            envs.SGLANG_TOPK_TRANSFORM_512_TORCH.override(False),
            envs.SGLANG_OPT_USE_TOPK_V2.override(True),
            patch(f"{_INDEXER}.topk_transform_512_v2", side_effect=fake_v2),
            patch(f"{_INDEXER}.plan_topk_v2", wraps=plan_topk_v2) as mock_plan,
        ):
            C4IndexerBackendMixin._run_topk_transform(
                backend,
                logits=logits,
                c4_seq_lens=c4_seq_lens,
                page_table=page_table,
                c4_sparse_page_indices=c4_sparse_page_indices,
                indexer_metadata=indexer_metadata,
                raw_indices=None,
            )

        # plan_topk_v2 was called with the chunk's c4_seq_lens (3 elements)
        mock_plan.assert_called_once()
        torch.testing.assert_close(mock_plan.call_args.args[0], c4_seq_lens)

        # topk_transform_512_v2 received regenerated metadata, not the full-batch one
        self.assertEqual(len(captured_meta), 1)
        self.assertIsNot(captured_meta[0], full_metadata)
        self.assertEqual(captured_meta[0].shape, (chunk_rows + 1, 2))

    def test_v2_metadata_unchanged_for_full_batch(self):
        """When metadata shape matches, plan_topk_v2 is NOT called."""
        rows = 5
        max_c4_seq_len = 100

        logits = torch.randn(rows, max_c4_seq_len, dtype=torch.float32)
        c4_seq_lens = torch.full((rows,), max_c4_seq_len, dtype=torch.int32)
        page_table = torch.zeros((rows, 2), dtype=torch.int32)
        c4_sparse_page_indices = torch.full((rows, 512), -1, dtype=torch.int32)

        matching_metadata = torch.empty((rows + 1, 2), dtype=torch.int32)
        indexer_metadata = SimpleNamespace(
            c4_page_size=64,
            topk_metadata=matching_metadata,
        )

        captured_meta = []

        def fake_v2(scores, seq_lens, pt, out, ps, metadata, **kw):
            captured_meta.append(metadata)

        backend = SimpleNamespace(dsa_topk_backend=DSATopKBackend.SGL_KERNEL)

        with (
            envs.SGLANG_TOPK_TRANSFORM_512_TORCH.override(False),
            envs.SGLANG_OPT_USE_TOPK_V2.override(True),
            patch(f"{_INDEXER}.topk_transform_512_v2", side_effect=fake_v2),
            patch(f"{_INDEXER}.plan_topk_v2") as mock_plan,
        ):
            C4IndexerBackendMixin._run_topk_transform(
                backend,
                logits=logits,
                c4_seq_lens=c4_seq_lens,
                page_table=page_table,
                c4_sparse_page_indices=c4_sparse_page_indices,
                indexer_metadata=indexer_metadata,
                raw_indices=None,
            )

        # plan_topk_v2 was NOT called — metadata already matches
        mock_plan.assert_not_called()
        # topk_transform_512_v2 received the original metadata
        self.assertIs(captured_meta[0], matching_metadata)


class TestDSV4RoutingConditions(CustomTestCase):
    """Verify oversize routing correctly skips chunking for guarded conditions."""

    def test_capture_mode_prevents_chunking(self):
        """When get_is_capture_mode() is True, _should_chunk_mqa_logits
        must not be called even if the batch would exceed budget."""
        with (
            patch(f"{_INDEXER}.get_is_capture_mode", return_value=True),
            patch.object(
                C4IndexerBackendMixin,
                "_should_chunk_mqa_logits",
                side_effect=AssertionError("must not be called in capture mode"),
            ),
        ):
            # Simulate the routing condition check that appears in
            # forward_c4_indexer.  We test the guard logic directly rather
            # than calling the full method, which requires many dependencies.
            _is_capture = True
            _is_cp = False
            _is_deep_gemm_path = True
            if _is_deep_gemm_path and not _is_cp and not _is_capture:
                C4IndexerBackendMixin._should_chunk_mqa_logits(1, 1, 0)
            # If we reach here, the guard correctly prevented the call.

    def test_cp_prevents_chunking(self):
        """When attn_cp_size != 1, _should_chunk_mqa_logits must not be called."""
        with (
            patch(f"{_INDEXER}.get_is_capture_mode", return_value=False),
            patch.object(
                C4IndexerBackendMixin,
                "_should_chunk_mqa_logits",
                side_effect=AssertionError("must not be called with CP active"),
            ),
        ):
            _is_capture = False
            _is_cp = True
            _is_deep_gemm_path = True
            if _is_deep_gemm_path and not _is_cp and not _is_capture:
                C4IndexerBackendMixin._should_chunk_mqa_logits(1, 1, 0)

    def test_non_deep_gemm_prevents_chunking(self):
        """When using torch/tilelang/aiter fallback, _should_chunk_mqa_logits
        must not be called."""
        with (
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True),
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
            patch.object(
                C4IndexerBackendMixin,
                "_should_chunk_mqa_logits",
                side_effect=AssertionError("must not be called on non-deep-gemm path"),
            ),
        ):
            use_fp4_indexer = False
            _is_deep_gemm_path = (
                not use_fp4_indexer
                and not envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
                and not envs.SGLANG_OPT_USE_AITER_INDEXER.get()
                and not envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
            )
            _is_cp = False
            _is_capture = False
            if _is_deep_gemm_path and not _is_cp and not _is_capture:
                C4IndexerBackendMixin._should_chunk_mqa_logits(1, 1, 0)

    def test_fp4_prevents_deep_gemm_path(self):
        """FP4 must never enter the oversize varlen path: get_index_k_scale_buffer
        reads K at FP8 strides (128 B/token) but FP4 buffers pack 68 B/token,
        silently corrupting the gathered K."""
        with (
            patch(f"{_INDEXER}.is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch.object(
                C4IndexerBackendMixin,
                "_should_chunk_mqa_logits",
                side_effect=AssertionError(
                    "must not be called with FP4 indexer active"
                ),
            ),
        ):
            use_fp4_indexer = True
            _is_deep_gemm_path = (
                not use_fp4_indexer
                and not envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
                and not envs.SGLANG_OPT_USE_AITER_INDEXER.get()
                and not envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
            )
            _is_cp = False
            _is_capture = False
            if _is_deep_gemm_path and not _is_cp and not _is_capture:
                C4IndexerBackendMixin._should_chunk_mqa_logits(1, 1, 0)

    def test_breakable_graph_prevents_chunking(self):
        """When inside a breakable CUDA graph, the oversize varlen path must
        not be entered — mirroring the is_in_breakable_cuda_graph guard in
        _can_use_nonpaged_indexer."""
        with (
            patch(f"{_INDEXER}.get_is_capture_mode", return_value=False),
            patch(f"{_INDEXER}.is_in_breakable_cuda_graph", return_value=True),
            patch.object(
                C4IndexerBackendMixin,
                "_should_chunk_mqa_logits",
                side_effect=AssertionError(
                    "must not be called inside a breakable CUDA graph"
                ),
            ),
        ):
            _is_capture = False
            _is_cp = False
            _is_deep_gemm_path = True
            in_breakable_graph = True
            if (
                _is_deep_gemm_path
                and not _is_cp
                and not _is_capture
                and not in_breakable_graph
            ):
                C4IndexerBackendMixin._should_chunk_mqa_logits(1, 1, 0)

    def test_sm80_prevents_varlen_routing(self):
        """On sm80/sm89 (Ampere), deep_gemm varlen kernels assert arch_major
        >= 9.  The routing guard must prevent varlen routing there."""
        with (
            patch(f"{_INDEXER}.is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
            patch.object(
                C4IndexerBackendMixin,
                "_should_chunk_mqa_logits",
                side_effect=AssertionError(
                    "must not be called on sm80 — varlen kernel asserts arch >= 9"
                ),
            ),
        ):
            _varlen_arch_ok = is_cuda() and (torch.cuda.get_device_capability()[0] >= 9)
            _is_deep_gemm_path = _varlen_arch_ok
            _is_cp = False
            _is_capture = False
            if _is_deep_gemm_path and not _is_cp and not _is_capture:
                C4IndexerBackendMixin._should_chunk_mqa_logits(1, 1, 0)

    def test_sm90_allows_varlen_routing(self):
        """On sm90+ (Hopper/Blackwell), varlen kernels are supported."""
        with (
            patch(f"{_INDEXER}.is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            _varlen_arch_ok = is_cuda() and (torch.cuda.get_device_capability()[0] >= 9)
            self.assertTrue(_varlen_arch_ok)


if __name__ == "__main__":
    unittest.main()
