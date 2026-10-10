"""Test legacy TVM-FFI and upstream pybind11 FP4 indexer signatures."""

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.attention.dsv4 import index_logits

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDsv4DeepGemmMqaApi(CustomTestCase):
    def setUp(self):
        super().setUp()
        index_logits._uses_upstream_mqa_api.cache_clear()
        self.addCleanup(index_logits._uses_upstream_mqa_api.cache_clear)

    def test_flat_width_is_not_passed_as_schedule_meta(self):
        for upstream in (True, False):
            with self.subTest(upstream=upstream):
                index_logits._uses_upstream_mqa_api.cache_clear()
                calls = []

                def native(
                    q, kv, weights, starts, ends, max_seqlen_k, schedule_meta=None
                ):
                    self.assertIsNone(schedule_meta)
                    calls.append((starts, ends, max_seqlen_k))
                    return torch.zeros(q[0].shape[0], max_seqlen_k)

                def legacy(q, kv, weights, starts, ends, clean_logits, max_seqlen_k):
                    self.assertIs(clean_logits, False)
                    calls.append((starts, ends, max_seqlen_k))
                    return torch.zeros(q[0].shape[0], max_seqlen_k)

                native.__doc__ = "max_seqlen_k: int, schedule_meta: Tensor | None"
                legacy.__doc__ = "clean_logits: bool, max_seqlen_k: int"
                module = ModuleType("deep_gemm")
                module.fp8_fp4_mqa_logits = native if upstream else legacy
                starts = torch.tensor([0, 4], dtype=torch.int32)
                lengths = torch.tensor([3, 4], dtype=torch.int32)
                with patch.dict(sys.modules, {"deep_gemm": module}):
                    tiles = list(
                        index_logits.flat_index_logits_tiles(
                            q=(torch.zeros(2, 1, 4), torch.zeros(2, 1)),
                            kv=(torch.zeros(8, 4), torch.zeros(8)),
                            weights=torch.ones(2, 1),
                            starts=starts,
                            lengths=lengths,
                            context_lengths=[3, 4],
                            budget_bytes=4096,
                        )
                    )
                self.assertEqual(len(tiles), 1)
                self.assertEqual(tiles[0][0], slice(0, 2))
                self.assertEqual(tiles[0][1].shape, (2, 4))
                self.assertEqual(len(calls), 1)
                torch.testing.assert_close(calls[0][0], starts)
                torch.testing.assert_close(calls[0][1], starts + lengths)
                self.assertEqual(calls[0][2], 4)

    def test_paged_omits_clean_logits_only_for_upstream(self):
        for upstream in (True, False):
            with self.subTest(upstream=upstream):
                index_logits._uses_upstream_mqa_api.cache_clear()
                calls = []

                def native(
                    q, kv, weights, lengths, pages, metadata, max_len, indices=None
                ):
                    self.assertIsNone(indices)
                    calls.append((lengths, max_len))
                    return torch.zeros(2, 4)

                def legacy(
                    q, kv, weights, lengths, pages, metadata, max_len, clean_logits
                ):
                    self.assertIs(clean_logits, False)
                    calls.append((lengths, max_len))
                    return torch.zeros(2, 4)

                module = ModuleType("deep_gemm")
                module.fp8_fp4_mqa_logits = lambda: None
                module.fp8_fp4_mqa_logits.__doc__ = (
                    "max_seqlen_k: int, schedule_meta: Tensor | None"
                    if upstream
                    else "clean_logits: bool, max_seqlen_k: int"
                )
                module.fp8_fp4_paged_mqa_logits = native if upstream else legacy
                with patch.dict(sys.modules, {"deep_gemm": module}):
                    result = index_logits.deep_gemm_fp4_paged_mqa_logits(
                        (torch.zeros(2, 1, 4), torch.zeros(2, 1)),
                        torch.zeros(1, 4),
                        torch.ones(2, 1),
                        torch.tensor([3, 4], dtype=torch.int64),
                        torch.zeros(2, 1, dtype=torch.int32),
                        object(),
                        4,
                    )
                self.assertEqual(result.shape, (2, 4))
                self.assertEqual(len(calls), 1)
                torch.testing.assert_close(
                    calls[0][0], torch.tensor([[3], [4]], dtype=torch.int32)
                )
                self.assertEqual(calls[0][1], 4)


if __name__ == "__main__":
    unittest.main()
