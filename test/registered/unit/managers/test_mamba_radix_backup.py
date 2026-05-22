import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (
    ScheduleBatch,
    set_mamba_track_indices_from_reqs,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class FakeMambaPool:
    def __init__(self):
        self.freed = []

    def fork_from(self, src_index: torch.Tensor) -> torch.Tensor:
        return torch.tensor([int(src_index[0]) + 1000], dtype=torch.int64)

    def free(self, index: torch.Tensor) -> None:
        self.freed.append(int(index[0]))


class FakeReq:
    def __init__(self, *, finished: bool = False, last_track_seqlen: int = 128):
        self.pending_radix_mamba_slot = torch.tensor([7], dtype=torch.int64)
        self.radix_mamba_backup_slot = None
        self.radix_mamba_backup_seqlen = None
        self.mamba_last_track_seqlen = last_track_seqlen
        self._finished = finished

    def finished(self) -> bool:
        return self._finished


class TestMambaRadixBackup(CustomTestCase):
    def _schedule_batch(self, mamba_pool: FakeMambaPool):
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.enable_overlap = True
        batch.req_to_token_pool = SimpleNamespace(mamba_pool=mamba_pool)
        return batch

    def _processor(self, mamba_pool: FakeMambaPool):
        processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
        object.__setattr__(
            processor, "req_to_token_pool", SimpleNamespace(mamba_pool=mamba_pool)
        )
        return processor

    def test_backup_pending_slot_before_overlap_track(self):
        pool = FakeMambaPool()
        req = FakeReq()
        batch = self._schedule_batch(pool)

        self.assertTrue(batch._maybe_backup_pending_radix_mamba_slot(req))

        self.assertEqual(int(req.pending_radix_mamba_slot[0]), 7)
        self.assertEqual(int(req.radix_mamba_backup_slot[0]), 1007)
        self.assertEqual(req.radix_mamba_backup_seqlen, 128)

    def test_stale_finished_batch_uses_backup_and_releases_track_slot(self):
        pool = FakeMambaPool()
        req = FakeReq(finished=True)
        req.radix_mamba_backup_slot = torch.tensor([1007], dtype=torch.int64)
        req.radix_mamba_backup_seqlen = 128
        processor = self._processor(pool)
        batch = SimpleNamespace(
            mamba_track_mask=torch.tensor([True]),
            mamba_track_indices=torch.tensor([7], dtype=torch.int64),
        )

        processor._finalize_radix_mamba_backup(req, use_backup_for_cache=True)
        processor._release_stale_radix_mamba_track_slot(req, batch, 0)

        self.assertEqual(int(req.pending_radix_mamba_slot[0]), 1007)
        self.assertEqual(req.mamba_last_track_seqlen, 128)
        self.assertIsNone(req.radix_mamba_backup_slot)
        self.assertEqual(pool.freed, [7])

    def test_current_tracked_batch_keeps_pending_slot_and_frees_backup(self):
        pool = FakeMambaPool()
        req = FakeReq(finished=True)
        req.radix_mamba_backup_slot = torch.tensor([1007], dtype=torch.int64)
        req.radix_mamba_backup_seqlen = 128
        processor = self._processor(pool)

        processor._finalize_radix_mamba_backup(req, use_backup_for_cache=False)

        self.assertEqual(int(req.pending_radix_mamba_slot[0]), 7)
        self.assertIsNone(req.radix_mamba_backup_slot)
        self.assertEqual(pool.freed, [1007])

    def test_first_stale_track_without_backup_releases_pending_slot(self):
        pool = FakeMambaPool()
        req = FakeReq(finished=True, last_track_seqlen=None)
        processor = self._processor(pool)
        batch = SimpleNamespace(
            mamba_track_mask=torch.tensor([True]),
            mamba_track_indices=torch.tensor([7], dtype=torch.int64),
        )

        processor._release_stale_radix_mamba_track_slot(req, batch, 0)

        self.assertIsNone(req.pending_radix_mamba_slot)
        self.assertEqual(pool.freed, [7])

    def test_possible_spec_boundary_uses_backup_and_keeps_tracking_enabled(self):
        pool = FakeMambaPool()
        req = FakeReq()
        batch = self._schedule_batch(pool)
        batch.device = torch.device("cpu")
        batch.reqs = [req]

        can_track = batch._maybe_backup_pending_radix_mamba_slot(req)
        set_mamba_track_indices_from_reqs(batch, [can_track])

        self.assertTrue(bool(batch.mamba_track_mask[0]))
        self.assertEqual(int(batch.mamba_track_indices[0]), 7)
        self.assertEqual(int(req.radix_mamba_backup_slot[0]), 1007)


if __name__ == "__main__":
    unittest.main()
