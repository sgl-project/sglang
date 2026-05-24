import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    ScheduleBatch,
    set_mamba_track_indices_from_slots,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
)

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class FakeServerArgs:
    mamba_track_interval = 128

    def enable_mamba_extra_buffer(self):
        return True


class FakeSpecAlgorithm:
    def __init__(self, is_none: bool):
        self._is_none = is_none

    def is_none(self) -> bool:
        return self._is_none

    def supports_spec_v2(self) -> bool:
        return True


class FakeMambaPool:
    def __init__(self, alloc_results=None):
        self.alloc_results = list(alloc_results or [1007])
        self.alloc_calls = 0
        self.freed = []

    def alloc(self, size: int) -> torch.Tensor:
        assert size == 1
        self.alloc_calls += 1
        result = self.alloc_results.pop(0) if self.alloc_results else 1007
        if result is None:
            return None
        return torch.tensor([result], dtype=torch.int64)

    def free(self, index: torch.Tensor) -> None:
        self.freed.append(int(index[0]))


class FakeReq:
    def __init__(self, *, finished: bool = False, last_track_seqlen: int = 128):
        self.pending_radix_mamba_slot = torch.tensor([7], dtype=torch.int64)
        self.mamba_last_track_seqlen = last_track_seqlen
        self.origin_input_ids = list(range(250))
        self.output_ids = list(range(7))
        self._finished = finished

    @property
    def seqlen(self) -> int:
        return len(self.origin_input_ids) + len(self.output_ids)

    def finished(self) -> bool:
        return self._finished


class TestMambaRadixTrackSlots(CustomTestCase):
    def _schedule_batch(self, mamba_pool: FakeMambaPool, *, overlap: bool = True):
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.enable_overlap = overlap
        batch.req_to_token_pool = SimpleNamespace(mamba_pool=mamba_pool)
        batch.device = torch.device("cpu")
        return batch

    def _processor(self, mamba_pool: FakeMambaPool):
        processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
        object.__setattr__(
            processor, "req_to_token_pool", SimpleNamespace(mamba_pool=mamba_pool)
        )
        return processor

    def _batch(self, track_slot: int, *, spec: bool = False):
        return SimpleNamespace(
            mamba_track_mask=torch.tensor([True]),
            mamba_track_indices=torch.tensor([track_slot], dtype=torch.int64),
            spec_algorithm=FakeSpecAlgorithm(is_none=not spec),
        )

    def test_spare_track_slot_keeps_pending_as_backup(self):
        pool = FakeMambaPool([1007])
        req = FakeReq()
        batch = self._schedule_batch(pool)

        track_slot = batch._prepare_overlap_radix_mamba_track_slot(req)

        self.assertEqual(int(track_slot[0]), 1007)
        self.assertEqual(int(req.pending_radix_mamba_slot[0]), 7)
        self.assertEqual(pool.alloc_calls, 1)

    def test_first_checkpoint_tracks_pending_detached(self):
        pool = FakeMambaPool()
        req = FakeReq(last_track_seqlen=None)
        batch = self._schedule_batch(pool)

        track_slot = batch._prepare_overlap_radix_mamba_track_slot(req)

        self.assertEqual(int(track_slot[0]), 7)
        self.assertIsNone(req.pending_radix_mamba_slot)
        self.assertEqual(pool.alloc_calls, 0)

    def test_spare_alloc_failure_tracks_pending_detached(self):
        pool = FakeMambaPool([None])
        req = FakeReq()
        batch = self._schedule_batch(pool)

        track_slot = batch._prepare_overlap_radix_mamba_track_slot(req)

        self.assertEqual(int(track_slot[0]), 7)
        self.assertIsNone(req.pending_radix_mamba_slot)
        self.assertEqual(pool.alloc_calls, 1)

    def test_set_track_indices_from_explicit_slots(self):
        pool = FakeMambaPool()
        req = FakeReq()
        batch = self._schedule_batch(pool)
        batch.reqs = [req]

        set_mamba_track_indices_from_slots(batch, [torch.tensor([1007])])

        self.assertTrue(bool(batch.mamba_track_mask[0]))
        self.assertEqual(int(batch.mamba_track_indices[0]), 1007)

    def test_stale_finished_batch_releases_spare_and_keeps_pending(self):
        pool = FakeMambaPool()
        req = FakeReq(finished=True)
        processor = self._processor(pool)

        processor._release_stale_radix_mamba_track_slot(
            req, self._batch(track_slot=1007), 0
        )

        self.assertEqual(int(req.pending_radix_mamba_slot[0]), 7)
        self.assertEqual(pool.freed, [1007])

    def test_stale_finished_batch_releases_detached_pending_track(self):
        pool = FakeMambaPool()
        req = FakeReq(finished=True)
        req.pending_radix_mamba_slot = None
        processor = self._processor(pool)

        processor._release_stale_radix_mamba_track_slot(req, self._batch(7), 0)

        self.assertIsNone(req.pending_radix_mamba_slot)
        self.assertEqual(pool.freed, [7])

    def test_current_boundary_adopts_spare_and_frees_old_pending(self):
        pool = FakeMambaPool()
        req = FakeReq()
        processor = self._processor(pool)
        result = SimpleNamespace(num_correct_drafts_per_req_cpu=None)

        with patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.get_global_server_args",
            return_value=FakeServerArgs(),
        ):
            processor._mamba_prefix_cache_update(
                req, self._batch(track_slot=1007), result, 0
            )

        self.assertEqual(int(req.pending_radix_mamba_slot[0]), 1007)
        self.assertEqual(req.mamba_last_track_seqlen, 256)
        self.assertEqual(pool.freed, [7])

    def test_possible_spec_boundary_without_actual_cross_frees_spare(self):
        pool = FakeMambaPool()
        req = FakeReq()
        req.output_ids = list(range(5))
        processor = self._processor(pool)
        result = SimpleNamespace(num_correct_drafts_per_req_cpu=[0])

        with patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.get_global_server_args",
            return_value=FakeServerArgs(),
        ):
            processor._mamba_prefix_cache_update(
                req, self._batch(track_slot=1007, spec=True), result, 0
            )

        self.assertEqual(int(req.pending_radix_mamba_slot[0]), 7)
        self.assertEqual(req.mamba_last_track_seqlen, 128)
        self.assertEqual(pool.freed, [1007])

    def test_possible_spec_boundary_alloc_failure_no_cross_restores_pending(self):
        pool = FakeMambaPool()
        req = FakeReq()
        req.output_ids = list(range(5))
        req.pending_radix_mamba_slot = None
        processor = self._processor(pool)
        result = SimpleNamespace(num_correct_drafts_per_req_cpu=[0])

        with patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.get_global_server_args",
            return_value=FakeServerArgs(),
        ):
            processor._mamba_prefix_cache_update(
                req, self._batch(7, spec=True), result, 0
            )

        self.assertEqual(int(req.pending_radix_mamba_slot[0]), 7)
        self.assertEqual(req.mamba_last_track_seqlen, 128)
        self.assertEqual(pool.freed, [])


if __name__ == "__main__":
    unittest.main()
