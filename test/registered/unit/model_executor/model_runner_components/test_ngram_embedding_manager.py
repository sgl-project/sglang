"""Unit tests for LongCat ngram token table updates."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ForwardMode  # noqa: E402
from sglang.srt.model_executor.model_runner_components.ngram_embedding_manager import (  # noqa: E402
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

    def test_decode_batch_drops_stale_chunked_prefill_mask(self):
        # An extend batch with a chunked req becomes the running decode batch
        # after the chunked req is filtered out; its old mask must not survive.
        manager = NgramEmbeddingManager(
            enabled=True, table=torch.zeros((8, 16), dtype=torch.int32), n=3
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            reqs=[object(), object()],
            ne_token_table=None,
            ne_skip_token_table_update=torch.tensor([False, False, True]),
        )

        manager.prepare_for_forward(batch, chunked_req=None)

        self.assertIs(batch.ne_token_table, manager.table)
        self.assertIsNone(batch.ne_skip_token_table_update)


if __name__ == "__main__":
    unittest.main()
