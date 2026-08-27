import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    _topk_from_pooled_history_logits_unfused,
    topk_from_pooled_history_logits,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKPoolMqaBackend(CustomTestCase):
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
