from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _worker(scalar):
    return SimpleNamespace(
        device="cpu",
        block_size=8,
        _draft_block_ids_buf=None,
        _draft_num_token_non_padded=scalar,
        _draft_num_token_non_padded_bs=None,
    )


def test_draft_token_count_reuses_storage_and_updates_only_for_batch_changes(
    monkeypatch,
):
    scalar = torch.empty((), dtype=torch.int32)
    fill = Mock(wraps=scalar.fill_)
    monkeypatch.setattr(scalar, "fill_", fill)
    worker = _worker(scalar)

    for bs, expected_updates in [(2, 1), (2, 1), (1, 2), (5, 3), (5, 3), (2, 4)]:
        DFlashWorkerV2._ensure_draft_block_buffers(worker, bs)
        assert worker._draft_num_token_non_padded is scalar
        assert scalar.item() == bs * 8
        assert fill.call_count == expected_updates
        assert worker._draft_block_ids_buf.shape[0] >= bs


def test_draft_buffers_do_not_add_a_token_count_when_disabled():
    worker = _worker(None)
    for bs in (2, 1, 4):
        DFlashWorkerV2._ensure_draft_block_buffers(worker, bs)
        assert worker._draft_num_token_non_padded is None
        assert worker._draft_num_token_non_padded_bs is None
