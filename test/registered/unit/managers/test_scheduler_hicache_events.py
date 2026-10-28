"""HiCache progress must not depend on whether a prefill batch is admitted."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.scheduler_assembly import init_mode_stages  # noqa: E402
from sglang.srt.managers.scheduler_components.bundles import Streams
from sglang.srt.managers.scheduler_components.engine_state import (
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.step_planner import StepPlanner
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.disagg_owner_shells import prefill_owner_shell
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerHiCacheEvents(unittest.TestCase):
    def setUp(self):
        self.calls = Mock()
        self.scheduler = s = partial_scheduler()
        self.scheduler.parts.mode_owners = ModeOwners()
        s.parts.lists = CoreLists()
        s.parts.counters = LaunchCounters()
        s.scheduler_stage_metrics = None
        s.parts.scripted_scheduler_hook = None
        enter_scope(self, published_topology())
        self._override_flags()
        self.storage_enabled = enter_scope(
            self,
            patch(
                "sglang.srt.managers.scheduler.hicache_storage_enabled",
                return_value=True,
            ),
        )
        s.parts.kv.tree_cache = SimpleNamespace(check_hicache_events=self.calls.drain)
        s.hub.request_intake = SimpleNamespace(
            process_storage_prefetch_retries=self.calls.retry,
            requeue_ready_grammars=Mock(),
        )
        s.process_pending_chunked_abort = Mock()
        s.parts.lists.chunked_req = None
        s.parts.pending = []
        s.parts.mode_owners.disagg_prefill = prefill_owner_shell(
            resolve_waiting_bootstrap=lambda waiting: waiting,
            process_prefill_chunk=lambda chunked_req, waiting, pending: (
                chunked_req,
                waiting,
            ),
        )
        s.parts.mode_owners.disagg_decode = None
        s.parts.mode_owners.dllm = None
        s.parts.mode_owners.hisparse_coordinator = None
        s.parts.dp_attn_adapter = Mock()
        s.parts.dp_attn_adapter.maybe_prepare_mlp_sync_batch.return_value = None
        s.hub.step_planner = Mock(spec=StepPlanner)
        s.parts.metrics_sink = SimpleNamespace(enable_fpm=False)
        self.running_batch = SimpleNamespace(
            is_prefill_only=False, is_empty=lambda: True
        )

    def _override_flags(self):
        get_context().override(
            "test", enable_hierarchical_cache=True, disable_overlap_schedule=True
        )

    def test_pd_prefill_drains_before_admission_even_with_empty_queue(self):
        s = self.scheduler
        s.hub.step_planner.get_new_batch_prefill = self.calls.admit
        s.hub.step_planner.get_new_batch_prefill.return_value = None
        s.parts.ngram_embedding_manager = Mock()
        s.parts.ngram_embedding_manager.prepare_for_forward.return_value = None
        s.parts.batch_result_processor = Mock()
        s.mode.stages = init_mode_stages(
            s.parts, s.loop_schedules(), lora_ids_in_use=set
        )
        slot = ScheduleSlot()
        for waiting_queue in ([], [SimpleNamespace(rid="pending_l3")]):
            with self.subTest(waiting=bool(waiting_queue)):
                s.parts.lists.waiting_queue = waiting_queue
                self.calls.reset_mock()
                for _ in range(2):
                    self.assertIsNone(s.schedule_step(slot))
                self.assertEqual(
                    self.calls.mock_calls,
                    [call.drain(), call.retry(), call.admit(slot)] * 2,
                )

    def test_unified_drains_when_prefill_is_deferred(self):
        s = self.scheduler
        s.parts.metrics_sink = SimpleNamespace(enable_fpm=False)
        s._abort_on_waiting_timeout = Mock()
        s._abort_on_running_timeout = Mock()
        s.parts.mode_owners.disagg_prefill = None
        s.parts.lists.chunked_req = None
        s.parts.require_mlp_sync = False
        s.hub.step_planner.should_defer_prefill = self.calls.defer
        s.hub.step_planner.should_defer_prefill.return_value = True
        s.hub.step_planner.get_new_batch_prefill = Mock()
        s.hub.step_planner.get_decode_step = Mock(return_value=None)
        s.hub.step_planner.arm_prefill_decode_interval = Mock()
        s.parts.ngram_embedding_manager = Mock()
        s.parts.ngram_embedding_manager.prepare_for_forward.return_value = None

        s.parts.batch_result_processor = Mock()

        s.mode.stages = init_mode_stages(
            s.parts, s.loop_schedules(), lora_ids_in_use=set
        )
        step = s.schedule_step(ScheduleSlot())

        self.assertIsNone(step)
        s.hub.step_planner.get_new_batch_prefill.assert_not_called()
        self.assertEqual(
            self.calls.mock_calls,
            [
                call.drain(),
                call.retry(),
                call.defer(self.running_batch),
            ],
        )

    def test_pp_prefill_drains_before_admission(self):
        s = self.scheduler
        enter_scope(self, published_topology(pp_size=2))
        self._override_flags()
        enter_scope(
            self,
            get_parallel().override(
                pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
                world_group=None,
                attn_tp_group=None,
            ),
        )
        s.parts.streams.device = None
        s.parts.streams.schedule_stream = s.parts.streams.forward_stream = (
            s.parts.streams.copy_stream
        ) = None
        s.parts.spec_algorithm = None
        s.parts.kv.req_to_token_pool = s.parts.kv.token_to_kv_pool_allocator = None
        s.parts.workers.tp_worker = s.parts.workers.model_worker = (
            s.parts.future_map
        ) = None
        s.parts.streams = Streams(
            device="cpu",
            device_module=SimpleNamespace(stream=lambda stream: None),
            schedule_stream=None,
            forward_stream=None,
            copy_stream=None,
            async_d2h_worker=None,
        )
        s.parts.pp_spec_forward = None
        s.mode.step_launcher = SimpleNamespace(run_batch=Mock())
        s.parts.recv_skipper = None
        s.parts.flags = EngineFlags()
        s.ingest_requests = Mock(return_value=[])
        s.parts.lists.waiting_queue = []
        s.disagg_prefill.bootstrap_queue = SimpleNamespace(pop_bootstrapped=lambda: [])
        s.disagg_prefill.poll_transfer_rids = Mock(return_value=[])
        enter_scope(
            self,
            patch(
                "sglang.srt.managers.scheduler_assembly.PPChannel",
                return_value=Mock(is_first_rank=True, is_last_rank=True),
            ),
        )
        s.hub.step_planner.get_new_batch_prefill = self.calls.admit
        # Stop the infinite loop at admission; no PP transport or GPU is needed.
        s.hub.step_planner.get_new_batch_prefill.side_effect = StopIteration

        s.parts.batch_result_processor = Mock()

        s.mode.stages = init_mode_stages(
            s.parts, s.loop_schedules(), lora_ids_in_use=set
        )
        with self.assertRaises(StopIteration):
            s.event_loop_pp()

        self.assertEqual(
            self.calls.mock_calls,
            [call.drain(), call.retry(), call.admit(s.pp_ring.slots[0].schedule_slot)],
        )


if __name__ == "__main__":
    unittest.main()
