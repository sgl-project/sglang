import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_assembly import init_mode_stages  # noqa: E402
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402
from sglang.test.scheduler_request_lists import request_registry_of

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Req:
    def __init__(self, rid: str, slot: int):
        self.rid = rid
        self.kv = SimpleNamespace(req_pool_idx=slot)
        self.finished_reason = None
        self.hisparse_staging = False

    def finished(self) -> bool:
        return self.finished_reason is not None


class TestHiSparseStagingList(unittest.TestCase):
    def test_launch_stages_and_prefill_commit_drops_finished(self):
        """A launched last chunk stages instead of decoding; a request its
        prefill commit finished leaves staging at that commit."""
        live, done = _Req("live", 0), _Req("done", 1)
        scheduler = partial_scheduler()
        scheduler.parts.mode_owners = ModeOwners()
        scheduler.parts.lists = CoreLists()
        scheduler.parts.lists.waiting_queue = [live, done]
        scheduler.parts.lists.decoding = []
        scheduler.parts.lists.chunked_req = None
        scheduler.mode.disaggregation_mode = DisaggregationMode.NULL
        scheduler.parts.mode_owners.disagg_prefill = None
        scheduler.parts.mode_owners.disagg_decode = None
        scheduler.parts.grammar_manager = SimpleNamespace(
            grammar_queue=[], abort_requests=None
        )
        scheduler.parts.mode_owners.dllm = None
        scheduler.parts.mode_owners.hisparse_coordinator = HiSparseCoordinator.__new__(
            HiSparseCoordinator
        )
        scheduler.hisparse_coordinator.staging = []
        scheduler.mode.stages = init_mode_stages(
            scheduler.parts, scheduler.loop_schedules()
        )
        step = SimpleNamespace(
            prefill_reqs=lambda: [live, done],
            chunked_req=None,
        )

        Scheduler.join_launched_step(scheduler, step, ScheduleSlot())
        self.assertEqual(scheduler.parts.lists.waiting_queue, [])
        self.assertEqual(scheduler.parts.lists.decoding, [])
        self.assertEqual(scheduler.hisparse_coordinator.staging, [live, done])

        done.finished_reason = object()
        Scheduler.leave_finished_reqs(scheduler)
        self.assertEqual(scheduler.hisparse_coordinator.staging, [live])
        registry = request_registry_of(scheduler)
        self.assertEqual(registry.collect_inflight_reqs(), [live])

    def test_ready_stops_at_a_prefill_not_committed(self):
        """Readiness is a prefix of staging: a request whose prefill has not
        committed yet has no copy, and the requests behind it wait."""
        first, uncommitted, behind = _Req("a", 0), _Req("b", 1), _Req("c", 2)
        copy_done = SimpleNamespace(query=lambda: True)
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.staging = [first, uncommitted, behind]
        coordinator._staging_copy_done = {0: copy_done, 2: copy_done}
        coordinator.tp_world_size = 1
        coordinator.alloc_device_buffer = Mock()
        coordinator._skip_first_backup = [False] * 3

        ready = HiSparseCoordinator.collect_ready_reqs(coordinator)

        self.assertEqual(ready, [first])
        self.assertEqual(coordinator.staging, [uncommitted, behind])
        self.assertEqual(coordinator._staging_copy_done, {2: copy_done})
        coordinator.alloc_device_buffer.assert_called_once_with(first)


if __name__ == "__main__":
    unittest.main()
