"""Unit tests for the DSV4 indexer's per-forward MQA-logits row chunking."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import (
    MQA_LOGITS_MIN_ROWS_PER_CHUNK,
    mqa_logits_budget_bytes,
    mqa_logits_row_bytes,
    mqa_logits_rows_per_chunk,
)
from sglang.srt.layers.attention.dsv4.indexer import topk_transform_pytorch_vectorized
from sglang.srt.layers.attention.dsv4.metadata import (
    PagedIndexerMetadata,
    iter_row_chunks,
    plan_indexer_row_chunks,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_DSA_UTILS = "sglang.srt.layers.attention.dsa.utils"
_METADATA = "sglang.srt.layers.attention.dsv4.metadata"

# issue #35201: 372K raw tokens -> 92992 c4 columns, padded to 93184 by DeepGEMM.
_ISSUE_C4_COLS = 92992
_ISSUE_ALIGNED_COLS = 93184


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
        # A budget below one row floors at the minimum chunk, not 1-row launches.
        self.assertEqual(
            mqa_logits_rows_per_chunk(
                num_rows=4096, row_bytes=row_bytes, budget_bytes=1
            ),
            MQA_LOGITS_MIN_ROWS_PER_CHUNK,
        )
        self.assertIsNone(
            mqa_logits_rows_per_chunk(
                num_rows=MQA_LOGITS_MIN_ROWS_PER_CHUNK,
                row_bytes=row_bytes,
                budget_bytes=1,
            )
        )

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
            patch(f"{_DSA_UTILS}.get_device_module", return_value=device_module),
            patch(f"{_DSA_UTILS}.get_schedule", return_value=schedule),
            patch(f"{_DSA_UTILS}.is_hip", return_value=False),
            patch(f"{_DSA_UTILS}.is_xpu", return_value=False),
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


if __name__ == "__main__":
    unittest.main()
