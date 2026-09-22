"""CPU coverage for the kpool indexer's MQA-logits row chunking.

Long-context prefill scores query rows in chunks under a free-memory budget.
Chunking must not change the result: every row lands where the single call
puts it, padding rows stay -1, and the request-indexed page table is passed
whole because page_table_row_index addresses it by request-pool ID.
"""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_KPOOL = "sglang.srt.layers.attention.dsa.dsa_indexer_kpool"
# Over the 8M-element skip threshold, so the budget decides.
_LARGE_ROWS, _LARGE_COLS = 16384, 229376


def _plan_chunks(num_rows, num_cols, budget_bytes, *, capture_mode=False):
    """Run the real chunk planner; returns (chunks, budget_query_mock)."""
    planner = IndexerKPool.__new__(IndexerKPool)
    with (
        patch(f"{_KPOOL}.mqa_logits_budget_bytes", return_value=budget_bytes) as budget,
        patch(f"{_KPOOL}.is_hip", return_value=False),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        patch(f"{_KPOOL}.capture_mode.is_capture_mode", capture_mode),
    ):
        chunks = planner._mqa_logits_row_chunks(
            num_rows=num_rows, num_cols=num_cols, device=torch.device("cuda", 0)
        )
    return chunks, budget


class TestKPoolMqaLogitsRowChunks(CustomTestCase):
    def test_small_matrices_skip_the_budget_query(self):
        chunks, budget = _plan_chunks(64, 1024, 1 << 20)
        self.assertEqual(chunks, [(0, 64)])
        budget.assert_not_called()

    def test_chunks_cover_every_row_once_and_fit_the_budget(self):
        budget_bytes = 6 << 30
        chunks, _ = _plan_chunks(_LARGE_ROWS, _LARGE_COLS, budget_bytes)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], _LARGE_ROWS)
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
            self.assertEqual(prev_end, next_start)
        for start, end in chunks:
            self.assertGreater(end, start)
            self.assertLessEqual((end - start) * _LARGE_COLS * 4, budget_bytes)

    def test_a_matrix_within_budget_runs_in_one_call(self):
        chunks, _ = _plan_chunks(_LARGE_ROWS, _LARGE_COLS, 64 << 30)
        self.assertEqual(chunks, [(0, _LARGE_ROWS)])

    def test_real_capture_keeps_one_call_without_a_budget_query(self):
        chunks, budget = _plan_chunks(
            _LARGE_ROWS, _LARGE_COLS, 6 << 30, capture_mode=True
        )
        self.assertEqual(chunks, [(0, _LARGE_ROWS)])
        budget.assert_not_called()

    def test_breakable_graph_replay_still_chunks(self):
        """get_is_capture_mode() is true throughout breakable-graph replay, but
        its eager breaks run this path for real and must stay budgeted."""
        # capture_mode resolves the flag in its own namespace.
        with patch(
            "sglang.srt.model_executor.runner_utils.capture_mode."
            "is_in_breakable_cuda_graph",
            return_value=True,
        ):
            from sglang.srt.model_executor.runner import get_is_capture_mode

            self.assertTrue(get_is_capture_mode())
            chunks, _ = _plan_chunks(_LARGE_ROWS, _LARGE_COLS, 6 << 30)
        self.assertGreater(len(chunks), 1)


def _row_chunks(rows_per_chunk, *, num_rows, **_):
    if rows_per_chunk is None:
        return [(0, num_rows)]
    return [
        (s, min(s + rows_per_chunk, num_rows))
        for s in range(0, num_rows, rows_per_chunk)
    ]


def _fake_deep_gemm():
    # Logits carry each row's weight, so a mis-offset q/weights slice shows up.
    return SimpleNamespace(
        fp8_mqa_logits=lambda q, kv, w, ks, ke, clean_logits: (
            w[:, :1].expand(q.shape[0], kv[0].shape[0]).clone()
        )
    )


def _encoding_topk(width, calls):
    """A _topk_from_kpool_logits stand-in whose rows identify their inputs.

    Each output row encodes the scored weight, pooled length and whichever
    request/offset index it was given, and out_rows pads with -1 like the
    real kernel wrapper.
    """

    def topk(logits, pool_lens, **kwargs):
        calls.append(kwargs)
        index = kwargs.get("page_table_row_index")
        if index is None:
            index = kwargs.get("topk_offsets")
        if index is None:
            index = torch.zeros_like(pool_lens)
        row_ids = (
            logits[:, 0].to(torch.int32) * 1_000_000
            + pool_lens.to(torch.int32) * 1_000
            + index.to(torch.int32)
        )
        result = row_ids.unsqueeze(1).expand(-1, width).contiguous()
        out_rows = kwargs.get("out_rows")
        if out_rows is not None and out_rows > result.shape[0]:
            padded = torch.full((out_rows, width), -1, dtype=torch.int32)
            padded[: result.shape[0]] = result
            result = padded
        return result

    return topk


def _expected_rows(*, pool_lens, index, width, pad_rows=0):
    """What _encoding_topk returns when row r is scored with its own inputs:
    weight r + 1 (see _fake_deep_gemm), pool_lens[r] and index[r]."""
    weight_ids = torch.arange(1, pool_lens.shape[0] + 1, dtype=torch.int32)
    row_ids = weight_ids * 1_000_000 + pool_lens * 1_000 + index
    rows = row_ids.to(torch.int32).unsqueeze(1).expand(-1, width)
    padding = torch.full((pad_rows, width), -1, dtype=torch.int32)
    return torch.cat([rows, padding])


def _backend(*, topk, rows_per_chunk, index_topk=4, index_kpool=4):
    backend = SimpleNamespace(
        index_topk=index_topk,
        index_kpool=index_kpool,
        alt_stream=None,
        _topk_from_kpool_logits=topk,
        _mqa_logits_row_chunks=partial(_row_chunks, rows_per_chunk),
        _get_index_k_read_buffer=lambda pool, layer_id: None,
        _fp8_mqa_logits=IndexerKPool._fp8_mqa_logits,
    )
    backend._kpool_topk_by_row_chunks = partial(
        IndexerKPool._kpool_topk_by_row_chunks, backend
    )
    return backend


def _fp8_zeros(*shape):
    return torch.zeros(shape, dtype=torch.uint8).view(torch.float8_e4m3fn)


class TestKPoolPlanChunking(CustomTestCase):
    """The batched plan path: many requests' query rows in one scoring pass."""

    NUM_REQ_SLOTS = 64
    TOPK = 4
    PAD_ROWS = 5

    def _drive(self, *, req_ids, rows_per_chunk, topk_method):
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
        # Padded batch: q_fp8 has rows past the plan's real ones.
        total_q = n_real + self.PAD_ROWS
        metadata = SimpleNamespace(
            attn_metadata=SimpleNamespace(
                kpool_extend_plan=plan,
                topk_indices_offset=torch.arange(total_q, dtype=torch.int32) + 100,
            ),
            topk_transform_method=topk_method,
        )
        weights = torch.arange(1, total_q + 1, dtype=torch.float32).reshape(
            total_q, 1, 1
        )
        calls = []
        backend = _backend(
            topk=_encoding_topk(self.TOPK, calls), rows_per_chunk=rows_per_chunk
        )
        with (
            # deep_gemm is bound only under `if is_cuda()`.
            patch(f"{_KPOOL}.deep_gemm", _fake_deep_gemm(), create=True),
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
                q_fp8=_fp8_zeros(total_q, 4),
                weights=weights,
                metadata=metadata,
            )
        return result, calls, page_table

    def test_every_chunk_gets_the_whole_request_indexed_table(self):
        # Noncontiguous, out-of-order slots: a sliced table resolves request 1
        # to row 17 once a 16-row chunk shifts its base.
        req_ids = [1, 9, 2, 40, 7, 3, 63, 0] * 4
        _, calls, page_table = self._drive(
            req_ids=req_ids, rows_per_chunk=16, topk_method=TopkTransformMethod.PAGED
        )
        self.assertEqual(len(calls), 2)
        for chunk_idx, call in enumerate(calls):
            self.assertIs(call["page_table"], page_table)
            start = chunk_idx * 16
            torch.testing.assert_close(
                call["page_table_row_index"],
                torch.tensor(req_ids[start : start + 16], dtype=torch.int32),
            )

    def test_chunked_rows_match_the_single_call_including_padding(self):
        req_ids = [5, 11, 2, 60, 9, 33, 1] * 4
        for method in (TopkTransformMethod.PAGED, TopkTransformMethod.RAGGED):
            with self.subTest(method=method):
                expected, one_call, _ = self._drive(
                    req_ids=req_ids, rows_per_chunk=None, topk_method=method
                )
                actual, chunked, _ = self._drive(
                    req_ids=req_ids, rows_per_chunk=5, topk_method=method
                )
                self.assertEqual(len(one_call), 1)
                self.assertGreater(len(chunked), 1)
                n_real = len(req_ids)
                index = (
                    torch.tensor(req_ids, dtype=torch.int32)
                    if method == TopkTransformMethod.PAGED
                    else torch.arange(n_real, dtype=torch.int32) + 100
                )
                # Every real row scored with its own inputs, then -1 padding.
                reference = _expected_rows(
                    pool_lens=torch.arange(1, n_real + 1, dtype=torch.int32),
                    index=index,
                    width=self.TOPK,
                    pad_rows=self.PAD_ROWS,
                )
                torch.testing.assert_close(expected, reference, rtol=0, atol=0)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def _causal_seq_lens(q_lens, seq_lens):
    # Extend row j of a request attends to its prefix plus rows 0..j.
    return torch.cat(
        [
            torch.arange(s - q + 1, s + 1, dtype=torch.int32)
            for q, s in zip(q_lens, seq_lens)
        ]
    )


class TestKPoolPerRequestChunking(CustomTestCase):
    """The per-request path chunks one request's rows at a q_slice offset."""

    TOPK = 4
    POOL = 4

    def _drive(self, *, q_lens, seq_lens, rows_per_chunk):
        token_nums = sum(q_lens)
        width = self.TOPK + self.POOL - 1
        forward_batch = SimpleNamespace(
            batch_size=len(q_lens),
            extend_seq_lens_cpu=list(q_lens),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
            req_pool_indices=torch.tensor([7, 3][: len(q_lens)], dtype=torch.int64),
        )
        seqlens_expanded = _causal_seq_lens(q_lens, seq_lens)
        metadata = SimpleNamespace(
            get_page_table_64=lambda: torch.zeros((len(q_lens), 1), dtype=torch.int32),
            get_seqlens_expanded=lambda: seqlens_expanded,
            topk_transform_method=TopkTransformMethod.RAGGED,
            attn_metadata=SimpleNamespace(
                topk_indices_offset=torch.arange(token_nums, dtype=torch.int32) + 100,
            ),
        )
        # Fully cached current K: the path scores it without a pool gather.
        cache = [
            (0, _fp8_zeros(s // self.POOL, 4), torch.zeros(s // self.POOL))
            for s in seq_lens
        ]
        weights = torch.arange(1, token_nums + 1, dtype=torch.float32).reshape(
            token_nums, 1
        )
        calls = []
        backend = _backend(
            topk=_encoding_topk(width, calls),
            rows_per_chunk=rows_per_chunk,
            index_topk=self.TOPK,
            index_kpool=self.POOL,
        )
        with (
            patch(f"{_KPOOL}.deep_gemm", _fake_deep_gemm(), create=True),
            patch(
                f"{_KPOOL}.get_token_to_kv_pool",
                return_value=SimpleNamespace(page_size=64),
            ),
            patch(f"{_KPOOL}._should_fuse_kpool_topk", return_value=True),
        ):
            result = IndexerKPool._get_topk_ragged_kpool(
                backend,
                forward_batch=forward_batch,
                layer_id=0,
                q_fp8=_fp8_zeros(token_nums, 4),
                weights=weights,
                metadata=metadata,
                extend_pooled_cache=cache,
            )
        return result, calls

    def test_a_long_request_after_another_matches_the_single_call(self):
        # The second request starts at q offset 5, so its chunks must slice
        # relative to that request, not to the batch.
        kwargs = dict(q_lens=[5, 19], seq_lens=[40, 96])
        expected, one_call = self._drive(rows_per_chunk=None, **kwargs)
        actual, chunked = self._drive(rows_per_chunk=4, **kwargs)
        self.assertEqual(len(one_call), 2)
        self.assertGreater(len(chunked), len(one_call))
        pool_lens = torch.div(
            _causal_seq_lens(kwargs["q_lens"], kwargs["seq_lens"]),
            self.POOL,
            rounding_mode="floor",
        )
        reference = _expected_rows(
            pool_lens=pool_lens,
            index=torch.arange(pool_lens.shape[0], dtype=torch.int32) + 100,
            width=self.TOPK + self.POOL - 1,
        )
        torch.testing.assert_close(expected, reference, rtol=0, atol=0)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
