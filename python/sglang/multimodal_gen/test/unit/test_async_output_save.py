import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.scheduler_client import (
    AsyncSchedulerClient,
    _set_async_output_save_hint_for_batch,
)


class TestAsyncOutputSaveHint(unittest.TestCase):
    def test_req_defaults_to_sync_output_save(self):
        req = Req(prompt="x")

        self.assertIs(req.allow_async_output_save, False)

    def test_scheduler_keeps_single_queued_req_sync(self):
        scheduler = object.__new__(Scheduler)
        req = Req(prompt="x")
        scheduler.waiting_queue = [(None, req, 0.0)]

        scheduler._enable_async_output_save_for_queued_burst()

        self.assertIs(req.allow_async_output_save, False)

    def test_scheduler_marks_entire_queued_burst_async(self):
        scheduler = object.__new__(Scheduler)
        reqs = [Req(prompt="a"), Req(prompt="b"), Req(prompt="c")]
        scheduler.waiting_queue = [(None, req, 0.0) for req in reqs]

        scheduler._enable_async_output_save_for_queued_burst()

        self.assertTrue(all(req.allow_async_output_save for req in reqs))

    def test_scheduler_marks_grouped_req_async(self):
        scheduler = object.__new__(Scheduler)
        reqs = [Req(prompt="a"), Req(prompt="b")]
        scheduler.waiting_queue = [(None, reqs, 0.0)]

        scheduler._enable_async_output_save_for_queued_burst()

        self.assertTrue(all(req.allow_async_output_save for req in reqs))

    def test_scheduler_client_hint_marks_nested_req_batch(self):
        reqs = [Req(prompt="a"), Req(prompt="b")]

        _set_async_output_save_hint_for_batch([reqs[0], [reqs[1]]], True)

        self.assertTrue(all(req.allow_async_output_save for req in reqs))

    def test_scheduler_client_marks_all_active_forward_batches(self):
        client = object.__new__(AsyncSchedulerClient)
        reqs = [Req(prompt="a"), Req(prompt="b")]
        client._active_forward_batches = [[reqs[0]], [reqs[1]]]

        client._mark_active_forward_batches_for_async_output_save()

        self.assertTrue(all(req.allow_async_output_save for req in reqs))

    def test_gpu_worker_requires_scheduler_hint(self):
        req = Req(
            prompt="x",
            save_output=True,
            return_file_paths_only=True,
            enable_frame_interpolation=False,
            enable_upscaling=False,
        )
        worker = object.__new__(GPUWorker)
        worker.server_args = SimpleNamespace(disagg_role=RoleType.MONOLITHIC)

        with patch(
            "sglang.multimodal_gen.runtime.managers.gpu_worker.current_platform.is_cpu",
            return_value=False,
        ):
            self.assertIs(worker._can_async_save_output_paths(req), False)
            req.allow_async_output_save = True
            self.assertIs(worker._can_async_save_output_paths(req), True)

    def test_gpu_worker_short_circuits_without_scheduler_hint(self):
        req = Req(
            prompt="x",
            save_output=True,
            return_file_paths_only=True,
            enable_frame_interpolation=False,
            enable_upscaling=False,
        )
        worker = object.__new__(GPUWorker)

        with patch(
            "sglang.multimodal_gen.runtime.managers.gpu_worker.current_platform.is_cpu",
            return_value=False,
        ) as is_cpu:
            self.assertIs(worker._can_async_save_output_paths(req), False)
            is_cpu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
