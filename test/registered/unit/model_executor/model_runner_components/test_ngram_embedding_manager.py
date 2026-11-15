"""Unit tests for LongCat ngram token table updates."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.model_runner_components.ngram_embedding_manager import (
    NgramEmbeddingManager,
    update_ngram_token_table_after_sampling,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_ngram_info(batch_size: int, skip_token_table_update=None):
    return SimpleNamespace(
        token_table=torch.full((8, 16), -1, dtype=torch.int32),
        out_column_starts=torch.empty(batch_size, dtype=torch.int32),
        out_req_lens=torch.empty(batch_size, dtype=torch.int32),
        skip_token_table_update=skip_token_table_update,
    )


class TestNgramTokenTableUpdate(CustomTestCase):
    def test_chunked_prefill_mask_skips_pseudo_next_token(self):
        info = _make_ngram_info(
            4, skip_token_table_update=torch.tensor([False, True, False, True])
        )
        next_token_ids = torch.tensor([101, 202, 303, 404], dtype=torch.int64)
        req_pool_indices = torch.tensor([3, 4, 5, 6], dtype=torch.int64)
        seq_lens = torch.tensor([11, 22, 33, 44], dtype=torch.int64)

        with patch(
            "sglang.srt.model_executor.model_runner_components.ngram_embedding_manager.update_token_table"
        ) as update_mock:
            updated = update_ngram_token_table_after_sampling(
                ngram_embedding_info=info,
                next_token_ids=next_token_ids,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                batch_size=4,
            )

        self.assertTrue(updated)
        update_mock.assert_called_once()
        kwargs = update_mock.call_args.kwargs
        self.assertIs(kwargs["ne_token_table"], info.token_table)
        self.assertTrue(
            torch.equal(kwargs["tokens"], torch.tensor([101, 303], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(kwargs["row_indices"], torch.tensor([3, 5], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                kwargs["column_starts"], torch.tensor([11, 33], dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(kwargs["req_lens"], torch.ones(2, dtype=torch.int32))
        )
        self.assertIsNone(kwargs["ignore_tokens"])

    def test_all_requests_masked_does_not_update_table(self):
        info = _make_ngram_info(2, skip_token_table_update=torch.tensor([True, True]))

        with patch(
            "sglang.srt.model_executor.model_runner_components.ngram_embedding_manager.update_token_table"
        ) as update_mock:
            updated = update_ngram_token_table_after_sampling(
                ngram_embedding_info=info,
                next_token_ids=torch.tensor([101, 202], dtype=torch.int64),
                req_pool_indices=torch.tensor([3, 4], dtype=torch.int64),
                seq_lens=torch.tensor([11, 22], dtype=torch.int64),
                batch_size=2,
            )

        self.assertFalse(updated)
        update_mock.assert_not_called()

    def test_unmasked_update_writes_all_sampled_tokens(self):
        info = _make_ngram_info(3)
        next_token_ids = torch.tensor([101, 202, 303], dtype=torch.int64)
        req_pool_indices = torch.tensor([3, 4, 5], dtype=torch.int64)
        seq_lens = torch.tensor([11, 22, 33], dtype=torch.int64)

        with patch(
            "sglang.srt.model_executor.model_runner_components.ngram_embedding_manager.update_token_table"
        ) as update_mock:
            updated = update_ngram_token_table_after_sampling(
                ngram_embedding_info=info,
                next_token_ids=next_token_ids,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                batch_size=3,
            )

        self.assertTrue(updated)
        update_mock.assert_called_once()
        kwargs = update_mock.call_args.kwargs
        self.assertIs(kwargs["ne_token_table"], info.token_table)
        self.assertTrue(
            torch.equal(
                kwargs["tokens"], torch.tensor([101, 202, 303], dtype=torch.int32)
            )
        )
        self.assertTrue(torch.equal(kwargs["row_indices"], req_pool_indices))
        self.assertTrue(torch.equal(kwargs["column_starts"], info.out_column_starts))
        self.assertTrue(torch.equal(kwargs["req_lens"], info.out_req_lens))
        self.assertTrue(
            torch.equal(
                info.out_column_starts, torch.tensor([11, 22, 33], dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(info.out_req_lens, torch.ones(3, dtype=torch.int32))
        )
        self.assertIsNone(kwargs["ignore_tokens"])


class TestNgramPrepareForForward(CustomTestCase):
    def test_prepare_for_forward_mixed_mode_updates_table(self):
        manager = NgramEmbeddingManager(
            enabled=True,
            table=torch.full((4, 32), -1, dtype=torch.int32),
            n=3,
        )
        req_prefill = SimpleNamespace(
            origin_input_ids=[10, 11, 12, 13],
            output_ids=[],
            kv=SimpleNamespace(req_pool_idx=1),
        )
        req_decode = SimpleNamespace(
            origin_input_ids=[20, 21, 22, 23, 24],
            output_ids=[],
            kv=SimpleNamespace(req_pool_idx=2),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.MIXED,
            reqs=[req_prefill, req_decode],
            prefix_lens=[0, 4],
            extend_lens=[4, 1],
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
            ne_token_table=None,
            ne_skip_token_table_update=None,
        )

        with patch(
            "sglang.srt.model_executor.model_runner_components.ngram_embedding_manager.update_token_table"
        ) as update_mock:
            result_batch = manager.prepare_for_forward(batch, chunked_req=None)

        self.assertIs(result_batch, batch)
        self.assertIs(batch.ne_token_table, manager.table)
        update_mock.assert_called_once()
        kwargs = update_mock.call_args.kwargs
        self.assertIs(kwargs["ne_token_table"], manager.table)
        # Prefill: start=0, extend_len=4 -> tokens [10, 11, 12, 13], col_start 0
        # Decode: start=4 >= n=3, extend_len=1 -> tokens [22, 23, 24], col_start 2
        self.assertTrue(
            torch.equal(
                kwargs["tokens"],
                torch.tensor([10, 11, 12, 13, 22, 23, 24], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(kwargs["row_indices"], torch.tensor([1, 2], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                kwargs["column_starts"], torch.tensor([0, 2], dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(kwargs["req_lens"], torch.tensor([4, 3], dtype=torch.int32))
        )

    def test_prepare_for_forward_prebuilt_mode_updates_table(self):
        manager = NgramEmbeddingManager(
            enabled=True,
            table=torch.full((4, 32), -1, dtype=torch.int32),
            n=3,
        )
        req = SimpleNamespace(
            origin_input_ids=[8801, 8802, 8803, 8804, 8805, 8806, 8807, 8808],
            output_ids=[8809],
            kv=SimpleNamespace(req_pool_idx=1),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.PREBUILT,
            reqs=[req],
            req_pool_indices=torch.tensor([1], dtype=torch.int64),
            ne_token_table=None,
        )

        with patch(
            "sglang.srt.model_executor.model_runner_components.ngram_embedding_manager.update_token_table"
        ) as update_mock:
            result_batch = manager.prepare_for_forward(batch, chunked_req=None)

        self.assertIs(result_batch, batch)
        self.assertIs(batch.ne_token_table, manager.table)
        update_mock.assert_called_once()
        kwargs = update_mock.call_args.kwargs
        self.assertIs(kwargs["ne_token_table"], manager.table)
        # Total length is 9 > n=3, trailing 3 tokens are [8807, 8808, 8809] starting at col 6
        self.assertTrue(
            torch.equal(
                kwargs["tokens"],
                torch.tensor([8807, 8808, 8809], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(kwargs["row_indices"], torch.tensor([1], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(kwargs["column_starts"], torch.tensor([6], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(kwargs["req_lens"], torch.tensor([3], dtype=torch.int32))
        )

    def test_prepare_for_forward_prebuilt_short_sequence(self):
        manager = NgramEmbeddingManager(
            enabled=True,
            table=torch.full((4, 32), -1, dtype=torch.int32),
            n=3,
        )
        req = SimpleNamespace(
            origin_input_ids=[101],
            output_ids=[102],
            kv=SimpleNamespace(req_pool_idx=2),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.PREBUILT,
            reqs=[req],
            req_pool_indices=torch.tensor([2], dtype=torch.int64),
            ne_token_table=None,
        )

        with patch(
            "sglang.srt.model_executor.model_runner_components.ngram_embedding_manager.update_token_table"
        ) as update_mock:
            manager.prepare_for_forward(batch, chunked_req=None)

        update_mock.assert_called_once()
        kwargs = update_mock.call_args.kwargs
        # Total length is 2 <= n=3, all tokens written starting at col 0
        self.assertTrue(
            torch.equal(
                kwargs["tokens"],
                torch.tensor([101, 102], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(kwargs["row_indices"], torch.tensor([2], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(kwargs["column_starts"], torch.tensor([0], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(kwargs["req_lens"], torch.tensor([2], dtype=torch.int32))
        )

    def test_slot_reuse_overwrites_stale_tokens(self):
        table = torch.full((4, 32), -1, dtype=torch.int32)
        # Populate slot 1 with stale tokens from request A
        table[1, 0:8] = torch.tensor(
            [7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108], dtype=torch.int32
        )
        manager = NgramEmbeddingManager(
            enabled=True,
            table=table,
            n=3,
        )

        def mock_apply_update(
            tokens,
            ne_token_table,
            row_indices,
            column_starts,
            req_lens,
            ignore_tokens=None,
        ):
            offset = 0
            for row, col, length in zip(
                row_indices.tolist(), column_starts.tolist(), req_lens.tolist()
            ):
                ne_token_table[row, col : col + length] = tokens[
                    offset : offset + length
                ]
                offset += length

        # Request B admitted to slot 1 via ForwardMode.PREBUILT
        req_b = SimpleNamespace(
            origin_input_ids=[8801, 8802, 8803, 8804, 8805, 8806, 8807, 8808],
            output_ids=[8809],
            kv=SimpleNamespace(req_pool_idx=1),
        )
        batch_b = SimpleNamespace(
            forward_mode=ForwardMode.PREBUILT,
            reqs=[req_b],
            req_pool_indices=torch.tensor([1], dtype=torch.int64),
            ne_token_table=None,
        )

        with patch(
            "sglang.srt.model_executor.model_runner_components.ngram_embedding_manager.update_token_table",
            side_effect=mock_apply_update,
        ):
            manager.prepare_for_forward(batch_b, chunked_req=None)

        # Context cells [6, 7, 8] read by decode step must hold B's tokens, not stale A tokens
        context_cells = manager.table[1, 6:9].tolist()
        self.assertEqual(context_cells, [8807, 8808, 8809])


if __name__ == "__main__":
    unittest.main()
