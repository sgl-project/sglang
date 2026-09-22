import sys
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    C4IndexerBackendMixin,
    topk_transform_pytorch_vectorized,
)
from sglang.srt.layers.attention.dsv4.metadata import (
    NonPagedIndexerPlan,
    PagedIndexerMetadata,
    iter_row_chunks,
    plan_indexer_row_chunks,
)
from sglang.srt.layers.attention.mqa_logits_utils import (
    MQA_LOGITS_MAX_BYTES_ROCM,
    mqa_logits_budget_bytes,
    mqa_logits_row_bytes,
    mqa_logits_rows_per_chunk,
    mqa_logits_should_chunk,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=17, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsv4.indexer"
_METADATA = "sglang.srt.layers.attention.dsv4.metadata"
_MQA_UTILS = "sglang.srt.layers.attention.mqa_logits_utils"

# issue #35201: 372K raw tokens -> 92992 c4 columns, padded to 93184 by DeepGEMM.
_ISSUE_C4_COLS = 92992
_ISSUE_ALIGNED_COLS = 93184


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
                compressed_page_size=64,
                page_table=torch.zeros((1, 1), dtype=torch.int32),
                compressed_seq_lens=torch.tensor([65], dtype=torch.int32),
                use_topk_v2=False,
                force_deep_gemm_metadata=True,
            )

        self.assertIs(metadata.deep_gemm_metadata, expected)
        deep_gemm.get_num_sms.assert_called_once_with()
        deep_gemm.get_paged_mqa_logits_metadata.assert_called_once()
        args = deep_gemm.get_paged_mqa_logits_metadata.call_args.args
        torch.testing.assert_close(args[0], torch.tensor([[65]], dtype=torch.int32))
        self.assertEqual(args[1:], (64, 1))
        jit_metadata.assert_not_called()

    def test_torch_fallback_skips_deep_gemm_and_ineligible_topk_plan(self):
        with (
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
            envs.SGLANG_OPT_USE_TOPK_V2.override(True),
            patch("sglang.kernels.ops.attention.dsv4.plan_topk_v2") as plan_topk_v2,
        ):
            metadata = PagedIndexerMetadata(
                page_size=256,
                compressed_page_size=64,
                page_table=torch.zeros((1, 1), dtype=torch.int32),
                compressed_seq_lens=torch.tensor([65], dtype=torch.int32),
                use_topk_v2=False,
            )

        self.assertIsNone(metadata.deep_gemm_metadata)
        plan_topk_v2.assert_not_called()
        self.assertEqual(metadata.topk_metadata.numel(), 0)

    def test_physical_page_size_controls_metadata_and_replay(self):
        planner = MagicMock(return_value=torch.zeros((1, 2), dtype=torch.int32))
        deep_gemm = SimpleNamespace(
            get_num_sms=MagicMock(return_value=1),
            get_paged_mqa_logits_metadata=planner,
        )
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            metadata = [
                PagedIndexerMetadata(
                    page_size=256,
                    compressed_page_size=page_size,
                    page_table=torch.zeros((1, 3), dtype=torch.int32),
                    compressed_seq_lens=torch.tensor([65], dtype=torch.int32),
                    use_topk_v2=False,
                    force_deep_gemm_metadata=True,
                )
                for page_size in (64, 32, 32)
            ]

        self.assertEqual(
            [call.args[1] for call in planner.call_args_list], [64, 32, 32]
        )
        self.assertEqual([m.max_compressed_seq_len for m in metadata], [192, 96, 96])
        self.assertEqual([m.max_seq_len for m in metadata], [768, 768, 768])
        with self.assertRaisesRegex(AssertionError, "compressed_page_size"):
            metadata[0].copy_(metadata[1])

        destination, source = metadata[1:]
        source.page_table.fill_(7)
        source.compressed_seq_lens.fill_(17)
        page_table_ptr = destination.page_table.data_ptr()
        destination.copy_(source)
        self.assertEqual(destination.page_table.data_ptr(), page_table_ptr)
        torch.testing.assert_close(destination.page_table, source.page_table)
        torch.testing.assert_close(
            destination.compressed_seq_lens, source.compressed_seq_lens
        )


class TestDSV4FlashInferTopK(CustomTestCase):
    def test_compact_page_transform_respects_fuse_topk(self):
        score_storage = torch.arange(160, dtype=torch.float32).reshape(2, 80)
        scores = score_storage[:, 1:65]
        self.assertFalse(scores.is_contiguous())

        seq_lens = torch.tensor([63, 64], dtype=torch.int32)
        page_tables = torch.tensor(
            [[7, 17, 8, 18], [11, 21, 12, 22]], dtype=torch.int32
        )[:, ::2]
        self.assertFalse(page_tables.is_contiguous())
        out_page_indices = torch.empty((2, 8), dtype=torch.int32)

        for fuse_topk, with_raw_output in product((False, True), repeat=2):
            with self.subTest(fuse_topk=fuse_topk, with_raw_output=with_raw_output):
                out_raw_indices = (
                    torch.empty_like(out_page_indices) if with_raw_output else None
                )

                def fake_top_k(input: torch.Tensor, k: int, **kwargs):
                    return torch.topk(
                        input,
                        k,
                        dim=-1,
                        largest=True,
                        sorted=kwargs["sorted"],
                    )

                top_k = MagicMock(side_effect=fake_top_k)
                top_k_page_table_transform = MagicMock()
                flashinfer = SimpleNamespace(
                    top_k=top_k,
                    top_k_page_table_transform=top_k_page_table_transform,
                )

                with (
                    patch.dict(sys.modules, {"flashinfer": flashinfer}),
                    envs.SGLANG_DSA_FUSE_TOPK.override(fuse_topk),
                    envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.override(True),
                    envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.override("small"),
                ):
                    backend = C4IndexerBackendMixin()
                    backend.flashinfer_topk_transform(
                        scores,
                        seq_lens,
                        page_tables,
                        out_page_indices,
                        page_size=64,
                        out_raw_indices=out_raw_indices,
                    )

                if not fuse_topk:
                    top_k_page_table_transform.assert_not_called()
                    top_k.assert_called_once()
                    call = top_k.call_args
                    self.assertTrue(call.args[0].is_contiguous())
                    self.assertEqual(call.args[0].shape, scores.shape)
                    self.assertEqual(call.args[1], out_page_indices.shape[1])
                    self.assertEqual(
                        call.kwargs,
                        {
                            "sorted": False,
                            "deterministic": True,
                            "tie_break": 1,
                            "dsa_graph_safe": True,
                        },
                    )
                    continue

                top_k.assert_not_called()
                top_k_page_table_transform.assert_called_once()
                call = top_k_page_table_transform.call_args
                self.assertIs(call.args[0], scores)
                self.assertIsNot(call.args[1], page_tables)
                self.assertTrue(call.args[1].is_contiguous())
                self.assertTrue(torch.equal(call.args[1], page_tables))
                self.assertIs(call.args[2], seq_lens)
                self.assertEqual(call.args[3], out_page_indices.shape[1])
                self.assertEqual(
                    call.kwargs,
                    {
                        "deterministic": True,
                        "tie_break": 1,
                        "dsa_graph_safe": True,
                        "page_size": 64,
                        "out": out_page_indices,
                        "out_raw_indices": out_raw_indices,
                    },
                )


class TestDSV4TopKDispatch(CustomTestCase):
    def test_v2_raw_output_uses_sparse_prefill_buffer_with_capture(self):
        page_table = torch.zeros((1, 1), dtype=torch.int32)
        compressed_seq_lens = torch.ones(1, dtype=torch.int32)
        page_indices = torch.full((1, 512), -1, dtype=torch.int32)
        raw_indices = torch.full_like(page_indices, -1)
        topk_metadata = torch.zeros((2, 2), dtype=torch.int32)

        indexer_metadata = object.__new__(PagedIndexerMetadata)
        indexer_metadata.page_size = 256
        indexer_metadata.compressed_page_size = 64
        indexer_metadata.page_table = page_table
        indexer_metadata.compressed_seq_lens = compressed_seq_lens
        indexer_metadata.topk_metadata = topk_metadata

        logits = torch.empty((1, 65), dtype=torch.float32)
        backend = C4IndexerBackendMixin()
        backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
        backend.token_to_kv_pool = SimpleNamespace(
            layer_mapping={0: SimpleNamespace(compress_layer_id=7)}
        )
        backend.forward_metadata = SimpleNamespace(
            indexer_metadata=indexer_metadata,
            core_metadata=SimpleNamespace(
                positions=torch.arange(1, dtype=torch.int64),
                page_table=page_table,
                c4_sparse_page_indices=page_indices,
                c4_sparse_raw_indices=raw_indices,
            ),
        )
        backend.hisparse_coordinator = None
        backend._forward_prepare_normal = MagicMock(
            return_value=(
                torch.empty((1, 1, 128)),
                torch.empty((1, 1, 1)),
            )
        )
        backend._get_nonpaged_indexer_plan = MagicMock(
            return_value=SimpleNamespace(query_rows=1, rows_per_chunk=None)
        )
        backend._gather_nonpaged_index_k = MagicMock(return_value=(object(), object()))
        backend._nonpaged_mqa_logits = MagicMock(return_value=logits)

        indexer_capturer = MagicMock()
        with (
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(True),
            patch(
                f"{_INDEXER}.get_global_indexer_capturer",
                return_value=indexer_capturer,
            ),
            patch(f"{_INDEXER}.topk_transform_paged") as topk_v1,
            patch(f"{_INDEXER}.topk_transform_paged_v2") as topk_v2,
        ):
            backend.forward_c4_indexer(
                x=torch.empty((1, 1)),
                q_lora=torch.empty((1, 1)),
                c4_indexer=SimpleNamespace(use_fp4_indexer=False, layer_id=0),
                forward_batch=SimpleNamespace(forward_mode=ForwardMode.EXTEND),
            )

        topk_v2.assert_called_once()
        args = topk_v2.call_args.args
        self.assertIs(args[0], logits)
        torch.testing.assert_close(args[1], compressed_seq_lens)
        torch.testing.assert_close(args[2], page_table)
        torch.testing.assert_close(args[3], page_indices)
        self.assertEqual(args[4], 64)
        self.assertIs(args[5], topk_metadata)
        self.assertEqual(args[6].data_ptr(), raw_indices.data_ptr())
        topk_v1.assert_not_called()
        indexer_capturer.capture.assert_called_once_with(7, raw_indices)


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
        metadata = SimpleNamespace(
            nonpaged_plan=None, compressed_page_size=64, mqa_logits_budget_bytes=None
        )
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
        torch.testing.assert_close(
            plan.ke, torch.tensor([0, 0, 0, 65], dtype=torch.int32)
        )
        torch.testing.assert_close(plan.gather_seq_lens, c4_seq_lens[-1:])

        metadata.nonpaged_plan = None
        batch.extend_seq_lens_cpu = [2, 2]
        with threshold.override(0):
            self.assertIsNone(build_plan())

    def test_plan_row_chunking_follows_the_forward_budget(self):
        backend = SimpleNamespace(_can_use_nonpaged_indexer=lambda **_: True)
        backend.dsa_topk_backend = SimpleNamespace(is_sgl_kernel=lambda: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False, index_topk=512)
        query_rows = 8192
        seq_len = 372_000
        batch = SimpleNamespace(
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            seq_lens_cpu=[seq_len],
            extend_seq_lens_cpu=[query_rows],
            extend_seq_lens=torch.tensor([query_rows], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_num_tokens=query_rows,
        )
        c4_seq_lens = torch.full((query_rows,), seq_len // 4, dtype=torch.int32)
        page_table = torch.zeros((query_rows, 1), dtype=torch.int32)

        def build_plan(budget):
            metadata = SimpleNamespace(
                nonpaged_plan=None,
                compressed_page_size=64,
                mqa_logits_budget_bytes=budget,
            )
            return C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                backend,
                c4_indexer=c4_indexer,
                forward_batch=batch,
                indexer_metadata=metadata,
                page_table=page_table,
                c4_seq_lens=c4_seq_lens,
                query_rows=query_rows,
            )

        # No budget measured this forward (small batch or graph): one call.
        self.assertIsNone(build_plan(None).rows_per_chunk)
        budget = 512 << 20
        plan = build_plan(budget)
        # 8192 rows x align256(93056) cols x 4 B is far over 512 MiB.
        self.assertIsNotNone(plan.rows_per_chunk)
        self.assertLess(plan.rows_per_chunk, query_rows)
        self.assertLessEqual(
            plan.rows_per_chunk * mqa_logits_row_bytes(plan.max_seqlen_k), budget
        )

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
        metadata = SimpleNamespace(
            nonpaged_plan=None, compressed_page_size=64, mqa_logits_budget_bytes=None
        )
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
        metadata = SimpleNamespace(
            nonpaged_plan=None, compressed_page_size=64, mqa_logits_budget_bytes=None
        )

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
            kv = C4IndexerBackendMixin._gather_nonpaged_index_k(
                c4_indexer=c4_indexer,
                token_to_kv_pool=token_to_kv_pool,
                plan=plan,
            )
            actual = C4IndexerBackendMixin._nonpaged_mqa_logits(
                q_indexer=q_indexer,
                weights=weights,
                kv=kv,
                plan=plan,
                rows=slice(0, plan.query_rows),
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


class TestMqaLogitsBudgetArithmetic(CustomTestCase):
    def test_row_bytes_follow_deepgemm_stride_alignment(self):
        # DeepGEMM pads the fp32 logits row stride to 1024 B, i.e. 256 columns.
        self.assertEqual(mqa_logits_row_bytes(1), 256 * 4)
        self.assertEqual(mqa_logits_row_bytes(256), 256 * 4)
        self.assertEqual(mqa_logits_row_bytes(257), 512 * 4)
        self.assertEqual(mqa_logits_row_bytes(_ISSUE_C4_COLS), _ISSUE_ALIGNED_COLS * 4)

    def test_rows_per_chunk_keeps_one_chunk_inside_budget(self):
        row_bytes = mqa_logits_row_bytes(_ISSUE_C4_COLS)
        budget = 512 << 20
        # 4096 rows x 93184 cols x 4 B = 1.42 GiB > 512 MiB: must slice.
        rows = mqa_logits_rows_per_chunk(
            num_rows=4096, row_bytes=row_bytes, budget_bytes=budget
        )
        self.assertIsNotNone(rows)
        self.assertLess(rows, 4096)
        self.assertLessEqual(rows * row_bytes, budget)
        # The whole matrix fits: single call.
        self.assertIsNone(
            mqa_logits_rows_per_chunk(
                num_rows=64, row_bytes=row_bytes, budget_bytes=budget
            )
        )
        # A tight budget never yields a chunk larger than the budget, even when
        # that means fewer rows than a full page of queries.
        tight = 64 << 20
        rows = mqa_logits_rows_per_chunk(
            num_rows=4096, row_bytes=row_bytes, budget_bytes=tight
        )
        self.assertEqual(rows, tight // row_bytes)
        self.assertLessEqual(rows * row_bytes, tight)
        # Few query rows still chunk when they do not fit.
        self.assertEqual(
            mqa_logits_rows_per_chunk(
                num_rows=100, row_bytes=row_bytes, budget_bytes=40 * row_bytes
            ),
            40,
        )
        # A budget below one row degrades to single-row chunks, never None.
        self.assertEqual(
            mqa_logits_rows_per_chunk(
                num_rows=4096, row_bytes=row_bytes, budget_bytes=1
            ),
            1,
        )

    def test_should_chunk_caps_the_budget_only_on_rocm(self):
        huge = 64 << 30
        # 16384 x 32768 x 4 B is exactly 2 GiB, aiter's compile-time ceiling.
        self.assertEqual(
            mqa_logits_should_chunk(
                num_rows=16384, num_cols=32768, get_budget_bytes=lambda: huge, rocm=True
            ),
            (True, MQA_LOGITS_MAX_BYTES_ROCM),
        )
        self.assertEqual(
            mqa_logits_should_chunk(
                num_rows=16384,
                num_cols=32768,
                get_budget_bytes=lambda: huge,
                rocm=False,
            ),
            (False, huge),
        )

    def test_should_chunk_skips_small_matrices_without_querying_the_budget(self):
        get_budget = MagicMock(return_value=1)
        # 64 decode rows x 100K columns is far below the 8M-element threshold.
        self.assertEqual(
            mqa_logits_should_chunk(
                num_rows=64, num_cols=100_000, get_budget_bytes=get_budget, rocm=False
            ),
            (False, 0),
        )
        get_budget.assert_not_called()

    def test_plan_combines_sm120_cap_with_budget(self):
        budget = 512 << 20
        by_budget = mqa_logits_rows_per_chunk(
            num_rows=8192,
            row_bytes=mqa_logits_row_bytes(_ISSUE_C4_COLS),
            budget_bytes=budget,
        )
        cases = (
            # (num_rows, num_cols, budget_bytes, sm120_row_cap) -> rows_per_chunk
            ((8192, 1024, None, None), None),
            ((8192, 1024, None, 4096), 4096),
            ((4096, 1024, None, 4096), None),
            ((8192, _ISSUE_C4_COLS, budget, None), by_budget),
            ((8192, _ISSUE_C4_COLS, budget, 4096), min(4096, by_budget)),
        )
        for (num_rows, num_cols, budget_bytes, cap), expected in cases:
            with self.subTest(num_rows=num_rows, num_cols=num_cols, cap=cap):
                self.assertEqual(
                    plan_indexer_row_chunks(
                        num_rows=num_rows,
                        num_cols=num_cols,
                        budget_bytes=budget_bytes,
                        sm120_row_cap=cap,
                    ),
                    expected,
                )

    def test_iter_row_chunks_covers_rows_exactly_once(self):
        self.assertEqual(
            list(iter_row_chunks(num_rows=10, rows_per_chunk=4)),
            [slice(0, 4), slice(4, 8), slice(8, 10)],
        )
        self.assertEqual(
            list(iter_row_chunks(num_rows=10, rows_per_chunk=None)), [slice(0, 10)]
        )
        self.assertEqual(
            list(iter_row_chunks(num_rows=10, rows_per_chunk=64)), [slice(0, 10)]
        )

    def test_static_budget_never_queries_free_memory(self):
        total = 80 << 30
        props = SimpleNamespace(total_memory=total)
        device_module = SimpleNamespace(
            get_device_properties=MagicMock(return_value=props)
        )
        schedule = SimpleNamespace(mem_fraction_static=0.9)
        with (
            envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.override(0.2),
            patch(f"{_MQA_UTILS}.get_device_module", return_value=device_module),
            patch(f"{_MQA_UTILS}.get_schedule", return_value=schedule),
            patch(f"{_MQA_UTILS}.is_hip", return_value=False),
            patch(f"{_MQA_UTILS}.is_xpu", return_value=False),
            patch(
                "torch.cuda.mem_get_info", return_value=(6 << 30, total)
            ) as mem_get_info,
        ):
            static = mqa_logits_budget_bytes(device_index=0, allow_sync=False)
            mem_get_info.assert_not_called()
            live = mqa_logits_budget_bytes(device_index=0, allow_sync=True)
            mem_get_info.assert_called_once_with(0)
        # static: 80 GiB x (1 - 0.9) x 0.2; live is further capped by 6 GiB free x 0.2.
        self.assertEqual(static, int(int(total * 0.1) * 0.2))
        self.assertEqual(live, int((6 << 30) * 0.2))


class TestPagedIndexerMetadataChunking(CustomTestCase):
    """The schedule list and the top-k plan list must be built over the exact
    row chunks the indexer loops over; a mismatch would silently score rows
    with another chunk's schedule."""

    def _build(self, *, num_rows: int, budget, use_topk_v2: bool):
        deep_gemm = SimpleNamespace(
            get_num_sms=MagicMock(return_value=1),
            get_paged_mqa_logits_metadata=MagicMock(
                side_effect=lambda c4, *_: torch.zeros((2, 2), dtype=torch.int32)
            ),
        )
        c4_seq_lens = torch.arange(1, num_rows + 1, dtype=torch.int32)
        page_table = torch.zeros(
            (num_rows, _ISSUE_ALIGNED_COLS // 64), dtype=torch.int32
        )
        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.override(False),
            patch(f"{_METADATA}.is_hip", return_value=False),
            patch(f"{_METADATA}.is_xpu", return_value=False),
            patch(f"{_METADATA}._IS_SM120", False),
            patch.object(
                PagedIndexerMetadata, "_mqa_logits_budget", return_value=budget
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.plan_topk_v2",
                side_effect=lambda seq_lens: seq_lens.new_zeros(
                    (seq_lens.shape[0] + 1, 2)
                ),
            ) as plan_topk_v2,
        ):
            metadata = PagedIndexerMetadata(
                page_size=256,
                compressed_page_size=64,
                page_table=page_table,
                compressed_seq_lens=c4_seq_lens,
                use_topk_v2=use_topk_v2,
            )
        return metadata, deep_gemm, plan_topk_v2

    def test_budget_splits_schedules_and_topk_plans_over_the_same_rows(self):
        num_rows, budget = 4096, 512 << 20
        metadata, deep_gemm, plan_topk_v2 = self._build(
            num_rows=num_rows, budget=budget, use_topk_v2=True
        )
        expected_rows = mqa_logits_rows_per_chunk(
            num_rows=num_rows,
            row_bytes=mqa_logits_row_bytes(_ISSUE_ALIGNED_COLS),
            budget_bytes=budget,
        )
        self.assertEqual(metadata.rows_per_chunk, expected_rows)
        self.assertEqual(metadata.mqa_logits_budget_bytes, budget)
        chunks = list(iter_row_chunks(num_rows=num_rows, rows_per_chunk=expected_rows))
        self.assertGreater(len(chunks), 1)

        self.assertIsInstance(metadata.deep_gemm_metadata, list)
        self.assertEqual(len(metadata.deep_gemm_metadata), len(chunks))
        schedule_rows = [
            call.args[0]
            for call in deep_gemm.get_paged_mqa_logits_metadata.call_args_list
        ]
        torch.testing.assert_close(
            torch.cat(schedule_rows), metadata.compressed_seq_lens.unsqueeze(-1)
        )
        self.assertEqual(
            [r.shape[0] for r in schedule_rows], [c.stop - c.start for c in chunks]
        )

        self.assertEqual(len(metadata.topk_metadata_chunks), len(chunks))
        # First call is the full-batch plan; the rest are one per chunk.
        plan_rows = [call.args[0] for call in plan_topk_v2.call_args_list]
        torch.testing.assert_close(plan_rows[0], metadata.compressed_seq_lens)
        torch.testing.assert_close(
            torch.cat(plan_rows[1:]), metadata.compressed_seq_lens
        )
        self.assertEqual(
            [r.shape[0] for r in plan_rows[1:]], [c.stop - c.start for c in chunks]
        )

    def test_no_budget_keeps_the_single_call_shape(self):
        metadata, deep_gemm, plan_topk_v2 = self._build(
            num_rows=4096, budget=None, use_topk_v2=True
        )
        self.assertIsNone(metadata.rows_per_chunk)
        self.assertIsNone(metadata.topk_metadata_chunks)
        self.assertIsInstance(metadata.deep_gemm_metadata, torch.Tensor)
        deep_gemm.get_paged_mqa_logits_metadata.assert_called_once()
        plan_topk_v2.assert_called_once()


class TestChunkedTopKMatchesUnchunked(CustomTestCase):
    """Each row's top-k depends only on its own logits row, sequence length and
    page-table row, so scoring the batch in row chunks must select the same
    pages as one pass. This is the property the chunk loop relies on."""

    def test_row_chunks_select_the_same_pages(self):
        torch.manual_seed(0)
        rows, width, topk, page_size = 37, 2048, 64, 64
        logits = torch.randn(rows, width, dtype=torch.float32)
        seq_lens = torch.randint(1, width, (rows,), dtype=torch.int32)
        page_table = torch.randint(
            0, 4096, (rows, width // page_size), dtype=torch.int32
        )

        def run(rows_per_chunk):
            out = torch.full((rows, topk), -1, dtype=torch.int32)
            for rows_slice in iter_row_chunks(
                num_rows=rows, rows_per_chunk=rows_per_chunk
            ):
                topk_transform_pytorch_vectorized(
                    logits[rows_slice],
                    seq_lens[rows_slice],
                    page_table[rows_slice],
                    out[rows_slice],
                    page_size,
                    None,
                )
            # Unsorted top-k: compare the selected sets row by row.
            return out.sort(dim=1).values

        expected = run(None)
        for rows_per_chunk in (1, 7, 16, rows - 1):
            with self.subTest(rows_per_chunk=rows_per_chunk):
                self.assertTrue(torch.equal(run(rows_per_chunk), expected))


class TestCandidateIndexerGating(CustomTestCase):
    def test_candidate_indexer_gating(self):
        from sglang.srt.layers.attention.dsv4 import candidate_indexer

        def platform(sm):
            return patch.object(
                candidate_indexer, "get_platform", lambda: SimpleNamespace(device_sm=sm)
            )

        flag = "sglang.srt.layers.deep_gemm_wrapper.configurer.DEEPGEMM_PAGED_SPARSE_MQA_LOGITS"
        # V4 models have no candidate source; Hopper selects through masks inline.
        with platform(100), patch(flag, True):
            self.assertIsNone(candidate_indexer.make_candidate_indexer(0, 8))
        with platform(90), patch(flag, False):
            self.assertIsNone(candidate_indexer.make_candidate_indexer(2048, 8))
        # Blackwell without DeepGEMM's sparse logits fails instead of falling back.
        with platform(100), patch(flag, False):
            with self.assertRaises(RuntimeError):
                candidate_indexer.make_candidate_indexer(2048, 8)


if __name__ == "__main__":
    unittest.main()
