"""CPU coverage for the kpool indexer's MQA-logits row chunking.

The kpool indexer scores query rows against pooled positions into one fp32
matrix that no pool sized by mem_fraction_static accounts for, so long-context
prefill chunks it by query rows under a free-memory budget. Two things must
hold for that loop to stay correct:

* the request-indexed page table is passed whole to every chunk, because
  `page_table_row_index` addresses it by absolute request-pool ID, and
* the budget still applies during breakable-graph replay, whose eager breaks
  run this path with live request metadata.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_KPOOL = "sglang.srt.layers.attention.dsa.dsa_indexer_kpool"


def _planner():
    """An IndexerKPool shell; __new__ skips an __init__ needing a device."""
    return IndexerKPool.__new__(IndexerKPool)


def _chunks(num_rows, num_cols, budget_bytes, capturing=False, capture_mode=False):
    with (
        patch(f"{_KPOOL}.mqa_logits_budget_bytes", return_value=budget_bytes),
        patch(f"{_KPOOL}.is_hip", return_value=False),
        patch(
            "torch.cuda.is_current_stream_capturing",
            return_value=capturing,
        ),
        patch(f"{_KPOOL}.capture_mode.is_capture_mode", capture_mode),
    ):
        return _planner()._mqa_logits_row_chunks(
            num_rows, num_cols, torch.device("cuda", 0)
        )


class TestKPoolMqaLogitsRowChunks(CustomTestCase):
    def test_small_matrices_run_in_one_call(self):
        # Below the 8M-element skip threshold: never worth a budget query.
        self.assertIsNone(_chunks(64, 1024, 1 << 20))

    def test_chunks_cover_every_row_once_and_fit_the_budget(self):
        # 16384 rows x 229376 pooled cols fp32 is the 14 GiB GLM-5.3-Flash
        # shape at a 917504-token prefill; 6 GiB is this host's real budget.
        num_rows, num_cols = 16384, 229376
        budget = 6 << 30
        chunks = _chunks(num_rows, num_cols, budget)
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 1)
        # Contiguous, non-overlapping, exact cover.
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], num_rows)
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
            self.assertEqual(prev_end, next_start)
        row_bytes = num_cols * 4
        for start, end in chunks:
            self.assertGreater(end, start)
            self.assertLessEqual((end - start) * row_bytes, budget)

    def test_a_budget_the_matrix_already_fits_runs_in_one_call(self):
        self.assertIsNone(_chunks(16384, 229376, 64 << 30))

    def test_real_capture_keeps_a_single_fixed_call(self):
        # Capture needs a fixed launch count, and mem_get_info would sync.
        self.assertIsNone(_chunks(16384, 229376, 6 << 30, capture_mode=True))
        self.assertIsNone(_chunks(16384, 229376, 6 << 30, capturing=True))

    def test_breakable_graph_replay_still_chunks(self):
        """get_is_capture_mode() is true throughout breakable-graph replay, but
        its eager breaks execute this path for real and must stay budgeted.

        capture_mode resolves is_in_breakable_cuda_graph in its own namespace,
        so that is where the replay flag has to be patched.
        """
        with patch(
            "sglang.srt.model_executor.runner_utils.capture_mode."
            "is_in_breakable_cuda_graph",
            return_value=True,
        ):
            # Guard against a guard that ignores the flag entirely.
            from sglang.srt.model_executor.runner import get_is_capture_mode

            self.assertTrue(get_is_capture_mode())
            chunks = _chunks(16384, 229376, 6 << 30)
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 1)


class TestKPoolChunkedPageTable(CustomTestCase):
    """The paged plan hands the indexer req_to_token itself plus one absolute
    request-pool ID per query row. Slicing that table would shift its base
    while the IDs keep addressing the original rows, so a later chunk would
    score another request's KV. This drives the real chunk loop."""

    NUM_REQ_SLOTS = 64
    TOPK = 4

    def _drive(self, *, req_ids, rows_per_chunk, topk_method):
        """Run _get_topk_ragged_kpool_plan, recording each chunk's arguments.

        rows_per_chunk=None exercises the unchunked call, which is the
        reference the chunked loop must reproduce.
        """
        n_real = len(req_ids)
        total_k_rows = 16
        page_table = torch.arange(self.NUM_REQ_SLOTS * 8, dtype=torch.int32).reshape(
            self.NUM_REQ_SLOTS, 8
        )
        plan = SimpleNamespace(
            seq_lens_expanded=torch.full((n_real,), 8, dtype=torch.int32),
            pooled_seq_lens_expanded=torch.arange(1, n_real + 1, dtype=torch.int32),
            ragged_q_ks=torch.zeros(n_real, dtype=torch.int32),
            ragged_q_ke=torch.full((n_real,), total_k_rows, dtype=torch.int32),
            ragged_total_k_rows=total_k_rows,
            ragged_k_u8=torch.zeros((total_k_rows, 4), dtype=torch.uint8),
            ragged_k_scale=torch.zeros((total_k_rows, 1), dtype=torch.float32),
            ragged_concat_page_table=torch.zeros(total_k_rows, dtype=torch.int32),
            ragged_paged_page_table=page_table,
            ragged_paged_page_table_row_index=torch.tensor(req_ids, dtype=torch.int32),
        )
        topk_offsets = torch.arange(n_real, dtype=torch.int32) * self.TOPK
        metadata = SimpleNamespace(
            attn_metadata=SimpleNamespace(
                kpool_extend_plan=plan, topk_indices_offset=topk_offsets
            ),
            topk_transform_method=topk_method,
        )

        calls = []

        def record(logits, pool_lens, **kwargs):
            calls.append({"pool_lens": pool_lens, **kwargs})
            return torch.zeros((logits.shape[0], self.TOPK), dtype=torch.int32)

        chunks = (
            None
            if rows_per_chunk is None
            else [
                (s, min(s + rows_per_chunk, n_real))
                for s in range(0, n_real, rows_per_chunk)
            ]
        )
        backend = SimpleNamespace(
            _topk_from_kpool_logits=record,
            _mqa_logits_row_chunks=lambda *a, **k: chunks,
            _get_index_k_read_buffer=lambda pool, layer_id: None,
            _fp8_mqa_logits=IndexerKPool._fp8_mqa_logits,
        )
        deep_gemm = SimpleNamespace(
            fp8_mqa_logits=lambda q, kv, w, ks, ke, clean_logits: torch.zeros(
                (q.shape[0], total_k_rows), dtype=torch.float32
            )
        )
        with (
            patch(f"{_KPOOL}.deep_gemm", deep_gemm),
            patch(f"{_KPOOL}.get_token_to_kv_pool", return_value=object()),
            patch(f"{_KPOOL}._should_fuse_kpool_topk", return_value=True),
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index."
                "gather_index_k_scale_prefix_into",
                lambda **kw: None,
            ),
        ):
            result = IndexerKPool._get_topk_ragged_kpool_plan(
                backend,
                forward_batch=None,
                layer_id=0,
                q_fp8=torch.zeros((n_real, 4), dtype=torch.uint8).view(
                    torch.float8_e4m3fn
                ),
                weights=torch.zeros((n_real, 1, 1), dtype=torch.float32),
                metadata=metadata,
            )
        return result, calls, page_table

    def test_every_chunk_gets_the_whole_request_indexed_table(self):
        # Noncontiguous, out-of-order slots: the failure mode is a chunk
        # resolving request 1 to row 17 after a 16-row base shift.
        req_ids = [1, 9, 2, 40, 7, 3, 63, 0] * 4
        _, calls, page_table = self._drive(
            req_ids=req_ids, rows_per_chunk=16, topk_method=TopkTransformMethod.PAGED
        )
        self.assertEqual(len(calls), 2)
        for chunk_idx, call in enumerate(calls):
            # Never sliced: the same full pool object each chunk.
            self.assertIs(call["page_table"], page_table)
            self.assertEqual(call["page_table"].shape[0], self.NUM_REQ_SLOTS)
            start = chunk_idx * 16
            torch.testing.assert_close(
                call["page_table_row_index"],
                torch.tensor(req_ids[start : start + 16], dtype=torch.int32),
            )

    def test_chunked_row_ids_match_the_unchunked_call(self):
        req_ids = [5, 11, 2, 60] * 8
        _, one_call, _ = self._drive(
            req_ids=req_ids, rows_per_chunk=None, topk_method=TopkTransformMethod.PAGED
        )
        _, chunked, _ = self._drive(
            req_ids=req_ids, rows_per_chunk=7, topk_method=TopkTransformMethod.PAGED
        )
        self.assertEqual(len(one_call), 1)
        self.assertGreater(len(chunked), 1)
        for key in ("page_table_row_index", "pool_lens", "seq_lens", "row_starts"):
            torch.testing.assert_close(
                torch.cat([c[key] for c in chunked]), one_call[0][key]
            )
        # Separate _drive calls build separate tables, so compare by value:
        # every chunk must still receive the whole pool, not a sliced view.
        for call in chunked:
            torch.testing.assert_close(call["page_table"], one_call[0]["page_table"])
            self.assertEqual(call["page_table"].shape[0], self.NUM_REQ_SLOTS)

    def test_query_indexed_offsets_are_sliced_per_chunk(self):
        """Without a row index the RAGGED path's tensors are per-query, so
        they must be sliced, unlike the request-indexed table."""
        req_ids = [0] * 16
        _, calls, _ = self._drive(
            req_ids=req_ids, rows_per_chunk=5, topk_method=TopkTransformMethod.RAGGED
        )
        self.assertGreater(len(calls), 1)
        self.assertTrue(all(c["page_table"] is None for c in calls))
        torch.testing.assert_close(
            torch.cat([c["topk_offsets"] for c in calls]),
            torch.arange(16, dtype=torch.int32) * self.TOPK,
        )

    def test_chunked_result_has_one_row_per_query(self):
        req_ids = [3, 8, 1, 55] * 4
        result, _, _ = self._drive(
            req_ids=req_ids, rows_per_chunk=6, topk_method=TopkTransformMethod.PAGED
        )
        self.assertEqual(result.shape, (len(req_ids), self.TOPK))


if __name__ == "__main__":
    unittest.main()
