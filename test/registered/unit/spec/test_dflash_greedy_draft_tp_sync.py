import types
import unittest

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SAMPLED = types.SimpleNamespace(is_all_greedy=False)


class _FakeTpSync:
    def __init__(self):
        self.calls = []

    def sync(self, site, values):
        self.calls.append((site, values.clone()))
        values.fill_(17)
        return values


def _worker(*, selector, is_domino=False):
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker._tp_sync = _FakeTpSync()
    worker.selector = selector
    worker._is_domino = is_domino
    worker._selector_sampling_enabled = True
    return worker


class TestDflashGreedyDraftTpSync(CustomTestCase):
    def _assert_synced(self, worker, draft_next, synced):
        assert len(worker._tp_sync.calls) == 1
        site, values = worker._tp_sync.calls[0]
        assert site == SpecTpSyncSite.DFLASH_DRAFT_GREEDY
        assert torch.equal(values, torch.tensor([[1, 2, 3]], dtype=torch.int64))
        assert torch.equal(synced, torch.full_like(draft_next, 17))

    def test_greedy_selector_draft_is_synchronized(self):
        worker = _worker(selector=object())
        draft_next = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        synced = worker._sync_greedy_draft(draft_next, None)
        self._assert_synced(worker, draft_next, synced)

    def test_sampled_batch_without_selector_is_synchronized(self):
        worker = _worker(selector=None)
        draft_next = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        synced = worker._sync_greedy_draft(draft_next, _SAMPLED)
        self._assert_synced(worker, draft_next, synced)

    def test_sampled_domino_draft_is_synchronized(self):
        worker = _worker(selector=object(), is_domino=True)
        draft_next = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        synced = worker._sync_greedy_draft(draft_next, _SAMPLED)
        self._assert_synced(worker, draft_next, synced)

    def test_sampled_selector_draft_is_not_resynchronized(self):
        worker = _worker(selector=object())
        sampling_info = types.SimpleNamespace(
            is_all_greedy=False, top_ks=torch.tensor([8, 8])
        )
        draft_next = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)

        synced = worker._sync_greedy_draft(draft_next, sampling_info)

        assert worker._tp_sync.calls == []
        assert synced is draft_next

    def test_mixed_selector_batch_syncs_only_greedy_rows(self):
        worker = _worker(selector=object())
        sampling_info = types.SimpleNamespace(
            is_all_greedy=False, top_ks=torch.tensor([1, 20])
        )
        draft_next = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)

        synced = worker._sync_greedy_draft(draft_next, sampling_info)

        # The full proposal is broadcast from rank 0 (the fake fills it with
        # 17), but only the greedy row takes rank 0's tokens; the sampled row
        # keeps its rank-local draw.
        assert len(worker._tp_sync.calls) == 1
        site, values = worker._tp_sync.calls[0]
        assert site == SpecTpSyncSite.DFLASH_DRAFT_GREEDY
        assert torch.equal(values, draft_next)
        assert torch.equal(
            synced, torch.tensor([[17, 17, 17], [4, 5, 6]], dtype=torch.int64)
        )
        # The rank-local proposal is not clobbered in place.
        assert torch.equal(
            draft_next, torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        )

    def test_selector_sampling_disabled_syncs_whole_tensor(self):
        worker = _worker(selector=object())
        worker._selector_sampling_enabled = False
        sampling_info = types.SimpleNamespace(
            is_all_greedy=False, top_ks=torch.tensor([1, 20])
        )
        draft_next = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)

        synced = worker._sync_greedy_draft(draft_next, sampling_info)

        # Selector sampling is disabled: every proposal is a rank-local argmax
        # regardless of requested top_k, so the whole tensor is synced from
        # rank 0 (the fake fills it with 17).
        assert len(worker._tp_sync.calls) == 1
        site, values = worker._tp_sync.calls[0]
        assert site == SpecTpSyncSite.DFLASH_DRAFT_GREEDY
        assert torch.equal(values, draft_next)
        assert torch.equal(synced, torch.full_like(draft_next, 17))


if __name__ == "__main__":
    unittest.main()
