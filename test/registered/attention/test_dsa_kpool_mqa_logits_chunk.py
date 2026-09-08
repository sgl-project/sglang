"""Unit tests for MQA-logits chunking in the DSA kpool indexer plan path.

Covers IndexerKPool._should_chunk_mqa_logits (budget arithmetic) and
IndexerKPool._get_topk_ragged_kpool_plan (chunked vs unchunked equivalence).

Regression test for the GLM-5.3-Flash long-context prefill OOM where
deep_gemm.fp8_mqa_logits allocated the full [extend_tokens, total_kv_rows]
fp32 logits buffer in one shot (e.g. 8192 x 348032 x 4B = 10.62 GiB) and
killed every TP rank.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import math
import torch
import unittest

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_GIB = 2**30
_POOL_SIZE = 4
_INDEX_TOPK = 2048  # group_topk = 2048 // 4 = 512 -> fused kpool topk path
_NUM_HEADS = 64
_HEAD_DIM = 128

# Mirrors the production crash: 8192 extend tokens x 348032 pooled kv rows.
_CRASH_NUM_Q = 8192
_CRASH_NUM_K = 348032


def _make_indexer() -> IndexerKPool:
    """Bypass __init__; the tested methods only need index_topk/index_kpool."""
    indexer = IndexerKPool.__new__(IndexerKPool)
    indexer.index_topk = _INDEX_TOPK
    indexer.index_kpool = _POOL_SIZE
    return indexer


def _fake_mqa_logits(
    q: torch.Tensor,
    kv: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Deterministic, chunk-invariant logits: unique value per (row, col).

    Row identity comes from ke (absolute per-row offsets survive chunk
    slicing), so the same q row always yields the same logits row no matter
    which chunk it is computed in.
    """
    k_fp8, _ = kv
    total_k = k_fp8.shape[0]
    cols = torch.arange(total_k, dtype=torch.float32, device=k_fp8.device)
    return ke.to(torch.float32).unsqueeze(1) * (total_k + 1) + cols.unsqueeze(0)


def _to_i32(values: List[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)


class TestShouldChunkMqaLogits(CustomTestCase):
    """Budget arithmetic of IndexerKPool._should_chunk_mqa_logits (no GPU)."""

    def _run(
        self, num_q: int, num_k: int, free_mem: int, total_mem: int
    ) -> Tuple[bool, int]:
        indexer = _make_indexer()
        with patch(
            "torch.cuda.mem_get_info", return_value=(free_mem, total_mem)
        ) as mock_info:
            result = indexer._should_chunk_mqa_logits(
                num_q, num_k, torch.device("cuda")
            )
        mock_info.assert_called_once()
        return result

    def test_small_matrices_skip_budget_entirely(self):
        indexer = _make_indexer()
        with patch("torch.cuda.mem_get_info") as mock_info:
            need_chunk, budget = indexer._should_chunk_mqa_logits(
                1000, 1000, torch.device("cpu")
            )
        self.assertEqual((need_chunk, budget), (False, 0))
        mock_info.assert_not_called()

    def test_production_crash_case_chunks_with_half_free_mem_budget(self):
        # The real OOM: logits = 8192 x 348032 x 4B = 10.62 GiB while only
        # ~14 GiB is free after static allocation on a 95 GiB card.
        need_chunk, budget = self._run(
            _CRASH_NUM_Q, _CRASH_NUM_K, free_mem=14 * _GIB, total_mem=95 * _GIB
        )
        self.assertTrue(need_chunk)
        self.assertEqual(budget, 7 * _GIB // 2)

        # The chunk loop must make progress and stay within the budget.
        bytes_per_row = _CRASH_NUM_K * 4
        max_rows = max(1, budget // bytes_per_row)
        self.assertGreaterEqual(max_rows, 1)
        self.assertLess(max_rows, _CRASH_NUM_Q)
        self.assertLessEqual(max_rows * bytes_per_row, budget)

    def test_plenty_of_free_memory_skips_chunking(self):
        need_chunk, _ = self._run(
            _CRASH_NUM_Q, _CRASH_NUM_K, free_mem=80 * _GIB, total_mem=95 * _GIB
        )
        self.assertFalse(need_chunk)

    def test_total_mem_cap_triggers_even_with_plenty_free(self):
        # logits = 8192 x 1310720 x 4B = 42.9 GiB > 30% of 95 GiB.
        total_mem = 95 * _GIB
        need_chunk, budget = self._run(
            _CRASH_NUM_Q, 1310720, free_mem=80 * _GIB, total_mem=total_mem
        )
        self.assertTrue(need_chunk)
        # min(80 GiB // 4, 30% of 95 GiB) = 20 GiB
        self.assertEqual(budget, 20 * _GIB)

    def test_zero_free_memory_still_yields_positive_budget(self):
        need_chunk, budget = self._run(
            _CRASH_NUM_Q, _CRASH_NUM_K, free_mem=0, total_mem=95 * _GIB
        )
        self.assertTrue(need_chunk)
        self.assertGreaterEqual(budget, 1)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestKpoolPlanChunkEquivalence(CustomTestCase):
    """Chunked and unchunked _get_topk_ragged_kpool_plan must agree bitwise."""

    POOL_LENS_PER_SEQ: List[int] = [600, 700]
    Q_LEN_PER_SEQ: List[int] = [4, 4]
    NUM_PAD_ROWS = 2

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")

        ks_list: List[int] = []
        ke_list: List[int] = []
        pool_len_list: List[int] = []
        seq_len_list: List[int] = []
        cu_k = 0
        for pooled, q_len in zip(self.POOL_LENS_PER_SEQ, self.Q_LEN_PER_SEQ):
            for j in range(q_len):
                # Row j of a sequence sees a growing pooled prefix; the last
                # row sees the whole sequence.
                visible = pooled - (q_len - 1 - j)
                ks_list.append(cu_k)
                ke_list.append(cu_k + visible)
                pool_len_list.append(visible)
                seq_len_list.append(visible * _POOL_SIZE)
            cu_k += pooled

        self.n_real = len(ks_list)
        self.total_k = cu_k
        self.total_q = self.n_real + self.NUM_PAD_ROWS

        self.plan = SimpleNamespace(
            seq_lens_expanded=_to_i32(seq_len_list, self.device),
            pooled_seq_lens_expanded=_to_i32(pool_len_list, self.device),
            ragged_q_ks=_to_i32(ks_list, self.device),
            ragged_q_ke=_to_i32(ke_list, self.device),
            ragged_total_k_rows=self.total_k,
            ragged_k_u8=torch.randint(
                0, 255, (self.total_k, _HEAD_DIM), dtype=torch.uint8
            ).to(self.device),
            ragged_k_scale=torch.rand(self.total_k, device=self.device) + 0.5,
            ragged_concat_page_table=torch.arange(
                (self.total_k + 63) // 64, dtype=torch.int32, device=self.device
            ),
            ragged_paged_page_table=None,
            ragged_paged_page_table_row_index=None,
        )
        self.metadata = SimpleNamespace(
            attn_metadata=SimpleNamespace(
                kpool_extend_plan=self.plan,
                topk_indices_offset=None,
            ),
            topk_transform_method=TopkTransformMethod.RAGGED,
        )
        self.q_fp8 = torch.randn(
            self.total_q, _NUM_HEADS, _HEAD_DIM, device=self.device
        ).to(torch.float8_e4m3fn)
        self.weights = torch.rand(self.total_q, _NUM_HEADS, 1, device=self.device) + 0.5

    def _run_plan(self, budget_bytes: Optional[int]) -> Tuple[torch.Tensor, int]:
        indexer = _make_indexer()
        # None keeps the real predicate, which skips chunking here via the
        # 8M-element static guard; otherwise force chunking with the budget.
        chunk_patcher = nullcontext()
        if budget_bytes is not None:
            chunk_patcher = patch.object(
                IndexerKPool,
                "_should_chunk_mqa_logits",
                return_value=(True, budget_bytes),
            )

        with (
            patch.object(
                IndexerKPool, "_get_index_k_read_buffer", return_value=MagicMock()
            ),
            patch(
                "sglang.srt.layers.attention.dsa.dsa_indexer_kpool"
                ".get_token_to_kv_pool",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index"
                ".gather_index_k_scale_prefix_into",
                return_value=None,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.deep_gemm"
            ) as mock_gemm,
            envs.SGLANG_DSA_FUSE_TOPK.override(False),
            chunk_patcher,
        ):
            mock_gemm.fp8_mqa_logits.side_effect = _fake_mqa_logits
            out = indexer._get_topk_ragged_kpool_plan(
                MagicMock(), 0, self.q_fp8, self.weights, self.metadata
            )
        return out, mock_gemm.fp8_mqa_logits.call_count

    @staticmethod
    def _assert_same_topk_set(
        test_case: unittest.TestCase, ref: torch.Tensor, out: torch.Tensor
    ) -> None:
        """The fused kpool top-k kernel orders results differently depending on
        batch size (two-stage binning), but selects the same set of indices.
        Compare per-row sorted values instead of raw order."""
        test_case.assertEqual(ref.shape, out.shape)
        for row in range(ref.shape[0]):
            ref_sorted = torch.sort(ref[row]).values
            out_sorted = torch.sort(out[row]).values
            test_case.assertTrue(
                torch.equal(ref_sorted, out_sorted),
                f"row {row}: selected index sets differ",
            )

    def test_chunked_matches_unchunked(self):
        ref, ref_calls = self._run_plan(None)
        self.assertEqual(ref_calls, 1)
        self.assertEqual(ref.shape[0], self.total_q)
        self.assertEqual(ref.dtype, torch.int32)
        # Padded rows beyond n_real must be the -1 sentinel.
        self.assertTrue(torch.all(ref[self.n_real :] == -1))

        for rows_per_chunk in (1, 2, 3):
            with self.subTest(rows_per_chunk=rows_per_chunk):
                budget = rows_per_chunk * self.total_k * 4
                out, calls = self._run_plan(budget)
                self.assertEqual(calls, math.ceil(self.n_real / rows_per_chunk))
                self._assert_same_topk_set(self, ref, out)

    def test_forced_chunk_with_huge_budget_runs_single_chunk(self):
        budget = self.n_real * self.total_k * 4
        out, calls = self._run_plan(budget)
        self.assertEqual(calls, 1)
        ref, _ = self._run_plan(None)
        self._assert_same_topk_set(self, ref, out)


if __name__ == "__main__":
    unittest.main()
