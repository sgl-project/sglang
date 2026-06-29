from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    C4IndexerBackendMixin,
)
from sglang.srt.layers.attention.dsv4.metadata import (
    NonPagedIndexerPlan,
    PagedIndexerMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV4NonPagedIndexer(unittest.TestCase):
    def _is_eligible(self, **overrides) -> bool:
        backend = SimpleNamespace(
            hisparse_coordinator=overrides.get("hisparse_coordinator")
        )
        c4_indexer = SimpleNamespace(
            use_fp4_indexer=overrides.get("use_fp4_indexer", False)
        )
        forward_batch = SimpleNamespace(
            forward_mode=overrides.get("forward_mode", ForwardMode.EXTEND),
            _original_forward_mode=overrides.get("original_forward_mode"),
            batch_size=overrides.get("batch_size", 1),
        )
        metadata = SimpleNamespace(
            use_prefill_cuda_graph=overrides.get("use_prefill_cuda_graph", False)
        )

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER,
                    "get",
                    return_value=overrides.get("feature_enabled", True),
                )
            )
            stack.enter_context(
                patch.object(
                    envs.SGLANG_OPT_USE_TILELANG_INDEXER,
                    "get",
                    return_value=overrides.get("use_tilelang", False),
                )
            )
            stack.enter_context(
                patch.object(
                    envs.SGLANG_OPT_USE_AITER_INDEXER,
                    "get",
                    return_value=overrides.get("use_aiter", False),
                )
            )
            stack.enter_context(
                patch.object(
                    envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH,
                    "get",
                    return_value=overrides.get("use_torch", False),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.layers.attention.dsv4.indexer.is_cuda",
                    return_value=overrides.get("is_cuda", True),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.layers.attention.dsv4.indexer.is_hip",
                    return_value=overrides.get("is_hip", False),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.layers.attention.dsv4.indexer.get_attention_cp_size",
                    return_value=overrides.get("cp_size", 1),
                )
            )
            stack.enter_context(
                patch(
                    "torch.cuda.is_current_stream_capturing",
                    return_value=overrides.get("is_capturing", False),
                )
            )
            return C4IndexerBackendMixin._can_use_nonpaged_indexer(
                backend,
                c4_indexer=c4_indexer,
                forward_batch=forward_batch,
                indexer_metadata=metadata,
            )

    def test_eligibility_is_fail_closed(self):
        self.assertTrue(self._is_eligible())
        cases = {
            "feature off": {"feature_enabled": False},
            "non CUDA": {"is_cuda": False},
            "HIP": {"is_hip": True},
            "decode": {"forward_mode": ForwardMode.DECODE},
            "mixed": {"forward_mode": ForwardMode.MIXED},
            "rewritten DP mode": {"original_forward_mode": ForwardMode.IDLE},
            "multi request": {"batch_size": 2},
            "prefill graph": {"use_prefill_cuda_graph": True},
            "FP4": {"use_fp4_indexer": True},
            "TileLang": {"use_tilelang": True},
            "AITER": {"use_aiter": True},
            "torch fallback": {"use_torch": True},
            "context parallel": {"cp_size": 2},
            "HiSparse": {"hisparse_coordinator": object()},
            "stream capture": {"is_capturing": True},
        }
        for label, overrides in cases.items():
            with self.subTest(label=label):
                self.assertFalse(self._is_eligible(**overrides))

    @staticmethod
    def _make_plan_inputs():
        query_rows = 4
        forward_batch = SimpleNamespace(
            # Deliberately not divisible by four: the plan must use floor C4
            # lengths while keeping a positive, page-aligned logits width.
            seq_lens=torch.tensor([262], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([262], dtype=torch.int32),
            extend_seq_lens_cpu=[query_rows],
            extend_seq_lens=torch.tensor([query_rows], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_num_tokens=query_rows,
        )
        metadata = SimpleNamespace(
            nonpaged_plan=None,
            c4_page_size=64,
        )
        page_table = torch.tensor([[3, 1]], dtype=torch.int32).repeat(query_rows, 1)
        c4_seq_lens = torch.tensor([62, 63, 64, 65], dtype=torch.int32)
        return forward_batch, metadata, page_table, c4_seq_lens, query_rows

    def test_builds_and_caches_single_request_plan(self):
        backend = SimpleNamespace(
            _can_use_nonpaged_indexer=MagicMock(return_value=True)
        )
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)
        forward_batch, metadata, page_table, c4_seq_lens, query_rows = (
            self._make_plan_inputs()
        )

        plan = C4IndexerBackendMixin._get_nonpaged_indexer_plan(
            backend,
            c4_indexer=c4_indexer,
            forward_batch=forward_batch,
            indexer_metadata=metadata,
            page_table=page_table,
            c4_seq_lens=c4_seq_lens,
            query_rows=query_rows,
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.seq_len_sum, 65)
        self.assertEqual(plan.max_seq_len, 65)
        self.assertEqual(plan.max_seqlen_k, 128)
        self.assertEqual(plan.query_rows, query_rows)
        torch.testing.assert_close(plan.page_table, page_table[:1])
        torch.testing.assert_close(
            plan.gather_seq_lens, torch.tensor([65], dtype=torch.int32)
        )
        torch.testing.assert_close(plan.ks, torch.zeros(query_rows, dtype=torch.int32))
        torch.testing.assert_close(plan.ke, c4_seq_lens)

        cached = C4IndexerBackendMixin._get_nonpaged_indexer_plan(
            backend,
            c4_indexer=c4_indexer,
            forward_batch=forward_batch,
            indexer_metadata=metadata,
            page_table=page_table,
            c4_seq_lens=c4_seq_lens,
            query_rows=query_rows,
        )
        self.assertIs(cached, plan)

    def test_plan_rejects_padded_or_multi_request_metadata(self):
        backend = SimpleNamespace(
            _can_use_nonpaged_indexer=MagicMock(return_value=True)
        )
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)

        for label, mutate in (
            ("query mismatch", lambda fb: setattr(fb, "extend_num_tokens", 5)),
            ("multi request", lambda fb: setattr(fb, "extend_seq_lens_cpu", [2, 2])),
            (
                "padded seq rows",
                lambda fb: setattr(fb, "seq_lens", torch.tensor([262, 1])),
            ),
            ("short prefix", lambda fb: setattr(fb, "seq_lens_cpu", torch.tensor([3]))),
        ):
            with self.subTest(label=label):
                forward_batch, metadata, page_table, c4_seq_lens, query_rows = (
                    self._make_plan_inputs()
                )
                mutate(forward_batch)
                plan = C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                    backend,
                    c4_indexer=c4_indexer,
                    forward_batch=forward_batch,
                    indexer_metadata=metadata,
                    page_table=page_table,
                    c4_seq_lens=c4_seq_lens,
                    query_rows=query_rows,
                )
                self.assertIsNone(plan)

    def test_nonpaged_dispatch_uses_gathered_kv_and_aligned_width(self):
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
        q_u8 = torch.zeros((query_rows, 2, 128), dtype=torch.uint8)
        q_indexer = q_u8.view(FP8_DTYPE)
        weights = torch.ones((query_rows, 2), dtype=torch.float32)
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
        torch.testing.assert_close(call.args[0], q_indexer)
        torch.testing.assert_close(call.args[1][0], k_u8.view(FP8_DTYPE))
        torch.testing.assert_close(
            call.args[1][1], scale_u8.view(torch.float32).squeeze(-1)
        )
        torch.testing.assert_close(call.args[2], weights)
        torch.testing.assert_close(call.args[3], plan.ks)
        torch.testing.assert_close(call.args[4], plan.ke)
        self.assertEqual(call.kwargs, {"clean_logits": False, "max_seqlen_k": 128})

    def test_forward_dispatch_selects_exactly_one_logits_path(self):
        query_rows = 4
        page_table = torch.tensor([[3], [3], [3], [3]], dtype=torch.int32)
        c4_seq_lens = torch.tensor([62, 63, 64, 65], dtype=torch.int32)
        core_metadata = SimpleNamespace(
            positions=torch.arange(query_rows, dtype=torch.int32),
            page_table=page_table,
            c4_sparse_page_indices=torch.full((query_rows, 2), -1, dtype=torch.int32),
        )
        indexer_metadata = object.__new__(PagedIndexerMetadata)
        indexer_metadata.page_size = 256
        indexer_metadata.page_table = page_table
        indexer_metadata.c4_seq_lens = c4_seq_lens
        indexer_metadata.deep_gemm_metadata = MagicMock()
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        c4_indexer = SimpleNamespace(layer_id=17, use_fp4_indexer=False)
        q_indexer = torch.zeros((query_rows, 2, 128), dtype=torch.uint8).view(FP8_DTYPE)
        weights_3d = torch.ones((query_rows, 2, 1), dtype=torch.float32)
        cache = torch.zeros((4, 64 * 132), dtype=torch.uint8)

        for use_nonpaged in (False, True):
            with self.subTest(use_nonpaged=use_nonpaged):
                plan = MagicMock(spec=NonPagedIndexerPlan) if use_nonpaged else None
                backend = SimpleNamespace(
                    token_to_kv_pool=MagicMock(),
                    forward_metadata=SimpleNamespace(
                        indexer_metadata=indexer_metadata,
                        core_metadata=core_metadata,
                    ),
                    debug_use_external_c4_sparse_indices=True,
                    hisparse_coordinator=None,
                    _forward_prepare_normal=MagicMock(
                        return_value=(q_indexer, weights_3d, cache)
                    ),
                    _get_nonpaged_indexer_plan=MagicMock(return_value=plan),
                    _forward_nonpaged_indexer=MagicMock(
                        return_value=torch.zeros((query_rows, 64))
                    ),
                )
                deep_gemm = SimpleNamespace(
                    fp8_paged_mqa_logits=MagicMock(
                        return_value=torch.zeros((query_rows, 64))
                    )
                )
                with (
                    patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                    patch.object(
                        envs.SGLANG_OPT_USE_TILELANG_INDEXER,
                        "get",
                        return_value=False,
                    ),
                    patch.object(
                        envs.SGLANG_OPT_USE_AITER_INDEXER,
                        "get",
                        return_value=False,
                    ),
                    patch.object(
                        envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH,
                        "get",
                        return_value=False,
                    ),
                ):
                    C4IndexerBackendMixin.forward_c4_indexer(
                        backend,
                        x=torch.zeros((query_rows, 1)),
                        q_lora=torch.zeros((query_rows, 1)),
                        c4_indexer=c4_indexer,
                        forward_batch=forward_batch,
                    )

                if use_nonpaged:
                    backend._forward_nonpaged_indexer.assert_called_once()
                    deep_gemm.fp8_paged_mqa_logits.assert_not_called()
                else:
                    backend._forward_nonpaged_indexer.assert_not_called()
                    deep_gemm.fp8_paged_mqa_logits.assert_called_once()

    def test_metadata_copy_drops_nonpaged_plan(self):
        def make_metadata(use_prefill_cuda_graph=False):
            metadata = object.__new__(PagedIndexerMetadata)
            metadata.page_size = 256
            metadata.page_table = torch.zeros((1, 1), dtype=torch.int32)
            metadata.c4_seq_lens = torch.ones(1, dtype=torch.int32)
            metadata.use_prefill_cuda_graph = use_prefill_cuda_graph
            metadata.deep_gemm_metadata = torch.zeros(1, dtype=torch.int32)
            metadata.topk_metadata = torch.zeros(1, dtype=torch.int32)
            metadata.nonpaged_plan = MagicMock()
            return metadata

        src = make_metadata()
        dst = make_metadata()
        with patch(
            "sglang.srt.layers.attention.dsv4.metadata.is_hip", return_value=False
        ):
            dst.copy_(src)
        self.assertIsNone(dst.nonpaged_plan)

        graph_metadata = make_metadata(use_prefill_cuda_graph=True)
        with (
            patch(
                "sglang.srt.layers.attention.dsv4.metadata.is_hip",
                return_value=False,
            ),
            self.assertRaises(AssertionError),
        ):
            dst.copy_(graph_metadata)


if __name__ == "__main__":
    unittest.main()
